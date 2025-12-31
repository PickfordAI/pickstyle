# train_vace_textmask.py

import argparse
import torch
import os
import wandb
import torch.backends.cudnn as cudnn
from glob import glob
import re

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import get_config
from dataset_style import VideoStyleDataset, video_collate_fn
from torch.utils.data import DataLoader, ConcatDataset
from vace.models.wan.configs import WAN_CONFIGS
from vace.wan.utils.utils import cache_video
import misc

from wan_vace_trainer import WanVace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--seed', default=0,
                        type=int, help='random seed')
    parser.add_argument(
        "--config",
        type=str,
        default="train/config/3stage/stage3.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="train-vace-anime",
        help="WANDB run name",
    )
    parser.add_argument('--disable-wandb', action='store_true')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser.parse_args()


def generate_sample(model, step, cap, src_video, src_mask):
    os.makedirs("train/vace_samples", exist_ok=True)
    out_video_path = f"train/vace_samples/{step}.mp4"

    # cap = "anime style."
    print("[INFO] Generating sample with caption:", cap)

    generated_video = model.generate(
        input_prompt=cap,
        input_frames=src_video,
        input_masks=src_mask,
        shift=5.0,
        sample_solver="dpm++",
        sampling_steps=20,
        t_guide=5,
        c_guide=4,
        n_prompt="",
        seed=2025,
        offload_model=False,
        uncond='text_context_separate',
    )
    cache_video(
        tensor=torch.stack([src_video, generated_video], dim=0),
        save_file=out_video_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    
    del generated_video
    return out_video_path


def train_model(vace_trainer, dataloader, optimizer, start_step, start_epoch, wandb_id, args, opts):
    n_nodes = misc.get_world_size() // torch.cuda.device_count()
    print(f"[INFO] Training on {n_nodes} nodes, {torch.cuda.device_count()} GPUs per node.")
    
    os.makedirs(args.train.save_lora_path, exist_ok=True)
    step = start_step
    n_data = len(dataloader.dataset)
    n_skip = start_step % (n_data // (torch.cuda.device_count() * n_nodes))
    skip_counter = 0
    for epoch in range(start_epoch, args.train.epochs):
        print(f"[INFO] Starting epoch {epoch + 1}/{args.train.epochs}")
        for batch in dataloader:
            if skip_counter < n_skip:
                print(f"[INFO] Skipping {skip_counter}/{n_skip}")
                skip_counter += 1
                continue
            # ----- STYLE STEP -----
            input_video, input_mask, target_video, caption = batch

            optimizer.zero_grad()

            iv = input_video[0].to(dtype=torch.bfloat16, device=vace_trainer.device)     # shape = [C, T, H, W]
            msk = input_mask[0].to(dtype=torch.bfloat16, device=vace_trainer.device)     # shape = [C, T, H, W]
            tv = target_video[0].to(dtype=torch.bfloat16, device=vace_trainer.device)    # shape = [C, T, H, W]
            text_prompt = caption[0]                       # string

            loss = vace_trainer.forward_train(
                text=text_prompt,
                src_video=iv,
                src_mask=msk,
                target_video=tv,
                use_gradient_checkpointing=args.model.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=args.model.use_gradient_checkpointing_offload,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vace_trainer.model_without_ddp.parameters(), 1.0)
            optimizer.step()
            
            print(f"[INFO] Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")
            log_info = {"loss": loss.item(), "epoch": epoch + 1}
            del iv, msk, tv, loss
            
            if misc.is_main_process():
                if step % args.train.sample_interval == 0 and args.train.sample_interval != -1:
                    if step > 0 or (step == 0 and args.train.sample_first_step):
                        """
                        Commented but ideally you have to use a video to see the style transfer quality through training
                        """
                        # try:
                        #     recon_batch = next(recon_iter)
                        # except (StopIteration, TypeError):
                        #     # restart the iterator if necessary
                        #     recon_iter = iter(recon_loader)
                        #     recon_batch = next(recon_iter)
                        # iv_r, msk_r, tv_r, cap_r = recon_batch
                        # iv_r = iv_r[0].to(dtype=torch.bfloat16, device=vace_trainer.device)
                        # msk_r = msk_r[0].to(dtype=torch.bfloat16, device=vace_trainer.device)
                        # text_prompt = "anime style."
                        # with torch.no_grad():
                        #     vace_trainer.model_without_ddp.activate_teacache()
                        #     sample_path = generate_sample(vace_trainer, step, text_prompt, iv_r, msk_r)
                        #     vace_trainer.model_without_ddp.deactivate_teacache()
                        
                        # # Clean up sample generation tensors
                        # del iv_r, msk_r, tv_r
                        
                        # log_info["generated_video"] = wandb.Video(
                        #     sample_path, caption="Generated Video", fps=16, format="mp4"
                        # )

                if not opts.disable_wandb:
                    wandb.log(log_info, step=step)

                if args.train.save_interval != -1 and step % args.train.save_interval == 0:
                    lora_weights_path = os.path.join(args.train.save_lora_path, f"lora_weights_step_{step}.pth")
                    training_info_path = os.path.join(args.train.save_lora_path, f"training_info_{step}.pth")
                    vace_trainer.save_lora_weights(lora_weights_path)
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'wandb_id': wandb_id,
                                'step': step + 1,
                                'epoch': epoch}, training_info_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                
            step += 1
            if step > args.train.end_step + 1:
                print(f"[INFO] Reached end step {args.train.end_step}. Ending training.")
                return

        print(f"[INFO] Finished epoch {epoch + 1}/{args.train.epochs}")
        

def create_multi_style_dataset(args):
    styles = args.image_dataset_v1.styles_list
    datasets = [VideoStyleDataset(style=style, **args.image_dataset_v1) for style in styles]
    for style in args.image_dataset_v2.styles_list:
        datasets.append(VideoStyleDataset(style=style, **args.image_dataset_v2))
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

def main():
    opts = parse_args()
    
    misc.init_distributed_mode(opts)
    misc.set_random_seed(opts.seed + misc.get_rank())
    cudnn.benchmark = True
    
    args = get_config(opts.config)

    dataset = create_multi_style_dataset(args)

    if opts.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=sampler, collate_fn=video_collate_fn, num_workers=4, pin_memory=False, persistent_workers=True,
    )
    print(f"[INFO] Style Dataset size: {len(dataset)}")

    print("[INFO] Initializing WanVaceTrainer model...")
    print(f"[INFO] Using model: {args.model.name}")
    print(f"[INFO] Using checkpoint: {args.model.ckpt_dir}")
    cfg = WAN_CONFIGS[args.model.name]
    vace_trainer = WanVace(
        cfg,
        opts,
        args.model.ckpt_dir,
        device_id=opts.gpu,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=args.model.use_usp,
        t5_cpu=False,
        dtype=torch.bfloat16
    )
    vace_trainer.model_without_ddp.teacache_init(
        use_ret_steps=True,
        teacache_thresh=0.2,
        sample_steps=20,
        ckpt_dir=args.model.ckpt_dir,
    )
    vace_trainer.model_without_ddp.deactivate_teacache()
    
    # Check for training info files with step numbers
    training_info_files = glob(f"{args.train.save_lora_path}/training_info_*.pth")
    should_resume = len(training_info_files) > 0
    if should_resume:
        # Find the latest training info file
        last_training_info_file = max(training_info_files, key=lambda f: int(re.search(r"training_info_(\d+)\.pth", os.path.basename(f)).group(1)))
        
        lora_ckpt_files = glob(f"{args.train.save_lora_path}/lora_weights_step_*.pth")
        last_lora_file = max(lora_ckpt_files, key=lambda f: int(re.search(r"lora_weights_step_(\d+)\.pth", os.path.basename(f)).group(1)))
        start_step = int(last_lora_file.split('_')[-1].split('.')[0]) + 1
        
        training_info = torch.load(last_training_info_file, map_location="cpu", weights_only=False)
        assert start_step == training_info['step'], (
            f"Last step in lora file {start_step} does not match step in training info {training_info['step']}"
        )
        start_epoch = training_info['epoch']
        
        print(f"[INFO] ⏩ Resuming. loading last checkpoint: {last_lora_file}, step: {start_step}")
        vace_trainer.add_lora_to_vace(pretrained_lora_path=last_lora_file, **args.model.lora_args)
    else:
        vace_trainer.add_lora_to_vace(**args.model.lora_args)
        start_step = 0
        start_epoch = 0
         
    
    block_type = args.model.lora_args.lora_target_modules.split("\\")[0]
    for name, param in vace_trainer.model_without_ddp.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, f"Found a non-LoRA param requiring grad: {name}"
            assert name.startswith(block_type), (
                f"LoRA param not under {block_type}: {name}"
            )
    print(f"✅ Only LoRA adapters are trainable, and all of them are in {block_type}.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, vace_trainer.model_without_ddp.parameters()),
        lr=args.train.lr,
    )
    if should_resume:
        optimizer.load_state_dict(training_info['optimizer_state_dict'])

    wandb_id = None
    if misc.is_main_process() and not opts.disable_wandb:
        if should_resume:
            wandb_id = training_info['wandb_id']
            wandb.init(id=wandb_id, project="VACE-Style", resume='must')
        else:
            wandb_id = wandb.util.generate_id()
            wandb.init(id=wandb_id, project="VACE-Style", config=args, name=opts.wandb_name)
    train_model(vace_trainer, dataloader, optimizer, start_step, start_epoch, wandb_id, args, opts)
    if misc.is_main_process() and not opts.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
