# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import gc
import math
import time
import random
import types
import logging
from contextlib import contextmanager
from functools import partial
from collections import OrderedDict

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from vace.wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                                 get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
from vace.models.wan.modules.model import VaceWanModel
from vace.models.utils.preprocessor import VaceVideoProcessor
from train.scheduler import FlowMatchScheduler
from peft import LoraConfig, inject_adapter_in_model


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
        v0: torch.Tensor,  # [B, C, T, H, W]
        v1: torch.Tensor,  # [B, C, T, H, W]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance(
    diff: torch.Tensor,  # [B, C, T, H, W]
    pred_cond: torch.Tensor,  # [B, C, T, H, W]
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 55,
):
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True)
        # print(f"diff_norm: {diff_norm}")
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return normalized_update


class WanVace(WanT2V):
    def __init__(
        self,
        config,
        opts,
        checkpoint_dir,
        device_id=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        dtype=torch.float32,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            dtype (`torch.dtype`, *optional*, defaults to torch.float32):
                Data type for model parameters.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.t5_cpu = t5_cpu
        self.dtype = dtype

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)
        
        # Cast model parameters to specified dtype
        self.model = self.model.to(dtype=self.dtype)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward,
                                                            usp_dit_forward_vace)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(
                usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.model_without_ddp = self.model
        self.distributed = opts.distributed
        if self.distributed:
            self.gpu = opts.gpu

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
                                           min_area=480 * 832,
                                           max_area=480 * 832,
                                           min_fps=self.config.sample_fps,
                                           max_fps=self.config.sample_fps,
                                           zero_start=True,
                                           seq_len=32760,
                                           keep_last=True)

        self.scheduler = FlowMatchScheduler(
            shift=5,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(config.num_train_timesteps, training=True)

    def vace_encode_frames(self, frames, ref_images=None, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def preprocess_mask(self, mask, vae_stride=None):
        vae_stride = self.vae_stride if vae_stride is None else vae_stride

        c, depth, height, width = mask.shape
        new_depth = int((depth + 3) // vae_stride[0])
        height = 2 * (int(height) // (vae_stride[1] * 2))
        width = 2 * (int(width) // (vae_stride[2] * 2))

        # reshape
        mask = mask[0, :, :, :]
        mask = mask.view(
            depth, height, vae_stride[1], width, vae_stride[1]
        )  # depth, height, 8, width, 8
        mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
        mask = mask.reshape(
            vae_stride[1] * vae_stride[2], depth, height, width
        )  # 1, 8*8, depth, height, width

        # interpolation
        mask = F.interpolate(mask.unsqueeze(0), size=(
            new_depth, height, width), mode='nearest-exact').squeeze(0)
        return mask

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            mask = self.preprocess_mask(mask, vae_stride)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    @torch.no_grad()
    def prepare_source(
        self,
        src_videos: list[torch.Tensor],
        src_masks: list[torch.Tensor],
        image_size: tuple[int, int],
        device: torch.device,
    ):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720 * 1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480 * 832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(
                f"image_size {image_size} is not supported")

        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_videos, src_masks)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_videos[i], src_masks[i], _, _, _ = self.vid_proc.load_video_pair(
                    sub_src_video, sub_src_mask)
                src_videos[i] = src_videos[i].to(device)
                src_masks[i] = src_masks[i].to(device)
                src_masks[i] = torch.clamp(
                    (src_masks[i][:1, :, :, :] + 1) / 2, min=0, max=1)
            else:
                src_videos[i], _, _, _ = self.vid_proc.load_video(
                    sub_src_video)
                src_videos[i] = src_videos[i].to(device)
                src_masks[i] = torch.ones_like(src_videos[i], device=device)

        return src_videos, src_masks

    def decode_latent(self, zs, ref_images=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)

        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)

        return vae.decode(trimed_zs)

    def encode_text_prompt(self, text):
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([text], self.device)
        else:
            context = self.text_encoder([text], torch.device('cpu'))
        context = [t.to(self.device, dtype=torch.float32) for t in context]
        return context

    def get_seq_len(self, video_shape):
        """
        Given a video shape (C, T, H, W), compute the sequence length
        that the VACE model expects in the DiT blocks. 
        Same logic as in WanVace.get_seq_len(...) but pulled into a small helper.
        """
        size = (video_shape[3], video_shape[2])  # (W, H)
        frame_num = video_shape[1]
        target_shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )
        return seq_len

    def save_lora_weights(self, save_path):
        state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param
        torch.save(state_dict, save_path)

    def add_lora_to_vace(
        self,
        lora_rank: int = 4,
        lora_alpha: int = 4,
        lora_target_modules: str = "qkv",
        init_lora_weights: str = "kaiming",
        pretrained_lora_path: str = None,
        adapter_name="default",
    ):
        """
        Injects a LoRA adapter into the entire VaceWanModel. Only the LoRA parameters
        will be trainable. Everything else remains frozen.
        """
        if init_lora_weights.lower() == "kaiming":
            init_lora_weights_flag = True
        else:
            init_lora_weights_flag = False

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights_flag,
            target_modules=lora_target_modules,
        )

        self.model = inject_adapter_in_model(
            lora_config, self.model, adapter_name=adapter_name)
        self.model.lora_scaling = lora_alpha / lora_rank

        for param in self.model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu], find_unused_parameters=False)
            self.model_without_ddp = self.model.module

        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = torch.load(
                pretrained_lora_path, map_location=self.device, weights_only=False)
            ckpt_has_module = any(k.startswith("module.") for k in state_dict)
            model_sd_keys = list(self.model.state_dict().keys())
            model_has_module = model_sd_keys[0].startswith("module.")
            if ckpt_has_module and not model_has_module:
                # drop leading "module."
                clean_sd = OrderedDict((k[len("module."):], v)
                                       for k, v in state_dict.items())
            elif not ckpt_has_module and model_has_module:
                # add leading "module."
                clean_sd = OrderedDict(
                    (f"module.{k}", v) for k, v in state_dict.items())
            else:
                clean_sd = state_dict

            missing_keys, unexpected_keys = self.model.load_state_dict(
                clean_sd, strict=False)
            assert unexpected_keys == [
            ], f"Unexpected keys found in the state_dict: {unexpected_keys}"
            print(
                f"[INFO] LoRA weights loaded successfully for adapter '{adapter_name}'.")
        else:
            print(f"[INFO] LoRA adapter '{adapter_name}' added successfully.")

        # Finally, move the entire model back to GPU (so both frozen + LoRA live on GPU)
        self.model.to(self.device)

    def forward_train(
        self,
        text: str,
        src_video: torch.Tensor,
        src_mask: torch.Tensor,
        target_video: torch.Tensor,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """
        A single training step:
          1) text -> encode -> context
          2) raw frames + masks -> VACE latents
          3) sample a random timestep
          4) add noise to the true latents
          5) call model.forward_vace(...) with that noisy latent & vace_context=z
          6) compute MSE loss against the true noise (weighted by scheduler training weight)

        Args:
          text:           (string)
          src_video:      tensor (C, T, H, W)
          src_mask:       tensor (C, T, H, W)
          target_video:   tensor [C, T, H, W]
        Returns:
          loss (scalar)
        """

        self.model.train()

        with torch.no_grad():
            # list of tensors -> [encoder_out, ...​]
            context = self.encode_text_prompt(text)
            context = [t.to(self.device, dtype=src_video.dtype) for t in context]
            # [tensor] -> [0] picks the tensor.
            target_video_latent = self.vae.encode([target_video])[0]
            target_video_latent = target_video_latent.to(self.device, dtype=src_video.dtype)

            # z0 = [C, T_latent, H_latent, W_latent] per frame
            z0 = self.vace_encode_frames([src_video], masks=[src_mask])
            m0 = self.vace_encode_masks([src_mask])
            z = self.vace_latent(z0, m0)
            z = [t.to(self.device, dtype=src_video.dtype) for t in z]
            # z is a list of latents, but in practice we only pass the first (unbatched) to forward_vace
            # so z[0] has shape (C//2, T_latent, H_latent, W_latent), concatenated inactive + reactive

        timestep_id = torch.randint(
            0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(self.device, dtype=src_video.dtype)

        noise = torch.randn_like(target_video_latent)
        noisy_latent = self.scheduler.add_noise(
            target_video_latent, noise, timestep)
        training_target = noise - target_video_latent

        arg_c = {"context": context,
                 "seq_len": self.get_seq_len(src_video.shape)}
        latent_model_input = [noisy_latent]

        noise_pred = self.model(
            latent_model_input,
            t=timestep,
            vace_context=z,
            vace_context_scale=1.0,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **arg_c
        )[0]  # [0] to extract the tensor from the returned tuple

        loss = F.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)

        return loss

    @torch.no_grad()
    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 t_guide=5.0,
                 c_guide=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=False,
                 use_apg=True,
                 uncond='text',
                 input_motions=None,
                 previous_clip_latents=None,
                 frame_stride=27,
                 input_ref_images=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            uncond (`str`, *optional*, defaults to 'text_context'):
                Options are 'text_context' or 'text'.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        # F = frame_num
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])
        #
        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size

        self.model.eval()

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            
        context = [t.to(device=self.device, dtype=input_frames.dtype) for t in context]
        context_null = [t.to(device=self.device, dtype=input_frames.dtype) for t in context_null]

        # vace context encode
        z0 = self.vace_encode_frames([input_frames], input_ref_images, masks=[input_masks])
        m0 = self.vace_encode_masks([input_masks], input_ref_images)
        z = self.vace_latent(z0, m0)
        z = [t.to(self.device, dtype=input_frames.dtype) for t in z]
        
        input_frames_null = torch.zeros_like(input_frames[0])
        null_input_ref_images = [[torch.zeros_like(input_ref_images[0][0])] * len(input_ref_images[0])] if input_ref_images is not None else input_ref_images
        
        if input_motions is not None:
            z2 = self.vace_latent(self.vace_encode_frames([input_motions], null_input_ref_images, masks=[input_masks]), m0)
            z2 = [t.to(self.device, dtype=input_frames.dtype) for t in z2]
        else:
            z2 = None
        
        z0_null = self.vace_encode_frames(
            [input_frames_null], null_input_ref_images, masks=[input_masks])
        z_null = self.vace_latent(z0_null, m0)
        z_null = [t.to(self.device, dtype=input_frames.dtype) for t in z_null]

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        
        noise = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=input_frames.dtype,
            device=self.device,
            generator=seed_g)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        # with amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}
        latents = noise

        # if uncond == 'text_context_separate':
        #     print(f"CFG with t_guide={t_guide}, c_guide={c_guide}")
        # else:
        #     if uncond == 'text_context':
        #         print(f"CFG with Text and Context t_guide={t_guide}")
        #     else:
        #         print(f"CFG with t_guide={t_guide}")
        
        if use_apg and uncond == 'text_context_separate':
            text_momentumbuffer  = MomentumBuffer(0.75) 
            context_momentumbuffer = MomentumBuffer(0.75) 
            
        if previous_clip_latents is not None:
            _, T_prev, _, _ = previous_clip_latents.shape
            noise_clip = torch.randn_like(previous_clip_latents).contiguous()
            noisy_latents_clip = self.scheduler.add_noise(previous_clip_latents, noise_clip, timesteps[0])
            latents[:, :T_prev, :, :] = noisy_latents_clip

        motion_context_scales = torch.exp(-torch.linspace(0, 3, len(timesteps), device=self.device, dtype=input_frames.dtype))
        for t_idx, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latents]
            timestep = [t]

            timestep = torch.stack(timestep).to(dtype=latents.dtype)
            motion_context_scale = motion_context_scales[t_idx].item()
                            
            self.model.to(self.device)
            if uncond == 'text':
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_motion=z2, vace_context_scale=context_scale, motion_vace_context_scale=motion_context_scale, **arg_c)[0]
                noise_pred_null = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_motion=z2, vace_context_scale=context_scale, motion_vace_context_scale=motion_context_scale, **arg_null)[0]
                noise_pred = noise_pred_null + \
                    t_guide * (noise_pred_cond - noise_pred_null)
            elif uncond == 'text_context_separate':
                
                noise_pred_null = self.model(
                    latent_model_input, t=timestep, vace_context=z_null, vace_context_scale=context_scale, motion_vace_context_scale=motion_context_scale, **arg_null)[0]
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_motion=z2, vace_context_scale=context_scale, motion_vace_context_scale=motion_context_scale, **arg_c)[0]
                noise_pred_null_text = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_motion=z2, vace_context_scale=context_scale, motion_vace_context_scale=motion_context_scale, **arg_null)[0]
                diff_uncond_text = noise_pred_cond - noise_pred_null_text
                diff_uncond_vace = noise_pred_null_text - noise_pred_null
                if use_apg:
                    noise_pred = noise_pred_cond + (t_guide - 1) * adaptive_projected_guidance(diff_uncond_text,
                                                                                                        noise_pred_cond,
                                                                                                        momentum_buffer=text_momentumbuffer,
                                                                                                        norm_threshold=55) \
                        + (c_guide - 1) * adaptive_projected_guidance(diff_uncond_vace,
                                                                                noise_pred_cond,
                                                                                momentum_buffer=context_momentumbuffer,
                                                                                norm_threshold=55)
                else:
                    noise_pred = noise_pred_null_text + t_guide *diff_uncond_text + c_guide * diff_uncond_vace
                del noise_pred_null, noise_pred_cond, noise_pred_null_text, diff_uncond_text, diff_uncond_vace
            elif uncond == 'text_context':
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_motion=z2, vace_context_scale=context_scale, motion_vace_context_scale=motion_context_scale, **arg_c)[0]
                noise_pred_null = self.model(
                    latent_model_input, t=timestep, vace_context=z_null, vace_context_motion=z_null, vace_context_scale=context_scale, motion_vace_context_scale=motion_context_scale, **arg_null)[0]
                noise_pred = noise_pred_null + \
                    t_guide * (noise_pred_cond - noise_pred_null)
            else:
                raise NotImplementedError(f"Unsupported uncond: {uncond}")

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents.unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = temp_x0.squeeze(0).to(dtype=input_frames.dtype)
            del noise_pred
            
            
            if previous_clip_latents is not None and t_idx < len(timesteps) - 1:
                _, T_prev, _, _ = previous_clip_latents.shape
                noise_clip = torch.randn_like(previous_clip_latents).contiguous()
                noisy_latents_clip = self.scheduler.add_noise(previous_clip_latents, noise_clip, timesteps[t_idx + 1])
                latents[:, :T_prev, :, :] = noisy_latents_clip

        x0 = [latents]
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        
        return videos[0]
