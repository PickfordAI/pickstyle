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
import traceback
from contextlib import contextmanager
from functools import partial
import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from peft import LoraConfig, inject_adapter_in_model

try:
    from vace.wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
    from vace.models.audio_analysis.wav2vec2 import Wav2Vec2Model
except ModuleNotFoundError:
    from wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
    from models.audio_analysis.wav2vec2 import Wav2Vec2Model
from .modules.vacetalk_model import VaceTalkWanModel
from ..utils.preprocessor import VaceVideoProcessor


class MomentumBuffer:
    def __init__(self, momentum: float): 
        self.momentum = momentum 
        self.running_average = 0 
    
    def update(self, update_value: torch.Tensor): 
        new_average = self.momentum * self.running_average 
        self.running_average = update_value + new_average
    


def project( 
        v0: torch.Tensor, # [B, C, T, H, W] 
        v1: torch.Tensor, # [B, C, T, H, W] 
        ): 
    dtype = v0.dtype 
    v0, v1 = v0.double(), v1.double() 
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4]) 
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1 
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance( 
          diff: torch.Tensor, # [B, C, T, H, W] 
          pred_cond: torch.Tensor, # [B, C, T, H, W] 
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
        print(f"diff_norm: {diff_norm}")
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm) 
        diff = diff * scale_factor 
    diff_parallel, diff_orthogonal = project(diff, pred_cond) 
    normalized_update = diff_orthogonal + eta * diff_parallel
    return normalized_update


def resize_and_centercrop(cond_image, target_size):
        """
        Resize image or tensor to the target size without padding.
        """

        # Get the original size
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size
        
        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)
        
        # Resize
        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous() 
            # crop
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size) 
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
            resized_image = np.array(resized_image)
            # tensor and crop
            resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor[:, :, None, :, :] 

        return cropped_tensor


class WanVaceTalk(WanT2V):
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        wav2vec_dir=None,
        t5_cpu=False,
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
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

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
        self.model = VaceTalkWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

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
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)
            
        # —– ENHANCE-A-VIDEO PATCH —–
        from .enhance_wan import inject_enhance_for_vace
        inject_enhance_for_vace(self.model)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=self.config.sample_fps,
            max_fps=self.config.sample_fps,
            zero_start=True,
            seq_len=32760,
            keep_last=True)
        
        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
        
    def add_lora_to_vace(
        self,
        lora_rank=4,
        lora_alpha=4,
        pretrained_lora_path=None,
    ):
        """
        Injects a LoRA adapter into the entire VaceWanModel. 
        """
        init_lora_weights_flag = True    
            
        lora_target_modules = r"vace_blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q|k|v)$"
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights_flag,
            target_modules=lora_target_modules,
        )

        self.model = inject_adapter_in_model(lora_config, self.model)

        state_dict = torch.load(pretrained_lora_path, map_location='cpu')
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        assert unexpected_keys == [], f"Unexpected keys found in the state_dict: {unexpected_keys}"
        print("LoRA weights loaded successfully.")

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
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
        mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)
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

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

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
    
    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def audio_prepare_single(self, audio_path, sample_rate=16000):
        ext = os.path.splitext(audio_path)[1].lower()
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = self.loudness_norm(human_speech_array, sr)
        return human_speech_array
        
    def get_audio_embedding(self, audio_path, sr=16000):
        human_speech = self.audio_prepare_single(audio_path)
        audio_duration = len(human_speech) / sr
        audio_duration = len(human_speech) / sr
        video_length = audio_duration * 25 # Assume the video fps is 25

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(human_speech, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()
        return audio_emb

    def generate(self,
                 input_prompt,
                 input_frames,
                 input_frames_unity,
                 input_masks_unity,
                 strength,
                 identity_strength,
                 input_masks,
                 input_ref_images,
                 size=(832, 480),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 text_guide_scale=5.0,
                 audio_guide_scale=4.0,
                 audio_path=None,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
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
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # vace context encode
        z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)
        
        input_frames_null = torch.zeros_like(input_frames)
        z0_null = self.vace_encode_frames(
            [input_frames_null], masks=[input_masks])
        z_null = self.vace_latent(z0_null, m0)
        
        assert os.path.exists(audio_path), f"Audio path {audio_path} does not exist."
        full_audio_embs = [self.get_audio_embedding(audio_path)]
        HUMAN_NUMBER = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + frame_num
        
        src_w, src_h = size
        background_mask = torch.ones([src_h, src_w])
        human_mask1 = torch.ones([src_h, src_w])
        human_mask2 = torch.ones([src_h, src_w])
        human_masks = [human_mask1, human_mask2, background_mask]
        
        ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
        lat_h, lat_w = target_shape[2], target_shape[3]        
        ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode='nearest').squeeze() 
        ref_target_masks = (ref_target_masks > 0) 
        ref_target_masks = ref_target_masks.float().to(self.device)
        
        audio_embs = []
        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        # split audio with window size
        for human_idx in range(HUMAN_NUMBER):   
            center_indices = torch.arange(
                audio_start_idx,
                audio_end_idx,
                1,
            ).unsqueeze(
                1
            ) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
            audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
            audio_embs.append(audio_emb)
        audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        noise = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        
        unity_latents = None
        if input_frames_unity is not None:
            unity_latents = self.vae.encode([input_frames_unity])[0] # (C, T, H, W)
            unity_T, noise_T = unity_latents.shape[1], noise.shape[1]
                
            if input_masks_unity is not None:
                unity_masks = self.preprocess_mask(input_masks_unity)
                unity_masks = torch.where(unity_masks > 0.5, 1.0, 0.0)
                
                """
                Turning 64 channel into 16 (same as latent and noise)
                """
                C, T, H, W = unity_masks.shape
                unity_mask_reshaped = unity_masks.reshape(C, T, H * W)
                unity_masks = F.interpolate(unity_mask_reshaped.unsqueeze(0).unsqueeze(0), size=(C // 4, T, H * W), mode='nearest-exact').squeeze().reshape(C // 4, T, H, W)
            else:
                unity_masks = torch.ones_like(noise)  # replace all
            
            if unity_T != noise_T: # When having ref image
                diff_t = noise_T - unity_T
                zero_pad = torch.zeros_like(unity_latents[:, :diff_t])
                unity_latents = torch.cat([zero_pad, unity_latents], dim=1)
                unity_masks = torch.cat([zero_pad, unity_masks], dim=1)
            

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

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

            if unity_latents is not None:
                if strength > 0:
                    start_timestep_idx = int((1 - strength) * len(timesteps))
                    start_timestep_unity_bg = timesteps[start_timestep_idx]
                    noisy_unity_bg = sample_scheduler.add_noise(unity_latents,
                                            noise,
                                            start_timestep_unity_bg.unsqueeze(0)
                                            )
                else:
                    start_timestep_unity_bg = 0
                
                if identity_strength > 0:
                    start_timestp_idx_id = int((1 - identity_strength) * len(timesteps))
                    start_timestep_unity_id = timesteps[start_timestp_idx_id]
                    noisy_unity_id = sample_scheduler.add_noise(unity_latents,
                                            noise,
                                            start_timestep_unity_id.unsqueeze(0)
                                            )
                else:
                    start_timestep_unity_id = 0

            arg_c = {'context': context, 'seq_len': seq_len, 'audio': audio_embs, 'ref_target_masks': ref_target_masks}
            arg_null_text = {'context': context_null, 'seq_len': seq_len, 'audio': audio_embs, 'ref_target_masks': ref_target_masks}
            arg_null = {'context': context_null, 'seq_len': seq_len, 'audio': torch.zeros_like(audio_embs)[-1:], 'ref_target_masks': ref_target_masks}
            latents = noise

            text_momentumbuffer  = MomentumBuffer(0.75) 
            audio_momentumbuffer = MomentumBuffer(0.75) 
            
            for _, t in enumerate(tqdm(timesteps)):
                if unity_latents is not None:
                    if t == start_timestep_unity_bg:
                        latents = torch.where(unity_masks == 1.0, noisy_unity_bg, latents)
                    if t == start_timestep_unity_id:
                        latents = torch.where(unity_masks == 0.0, noisy_unity_id, latents)
                latent_model_input = [latents]
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[0]
                noise_pred_drop_text = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,**arg_null_text)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, vace_context=z_null, vace_context_scale=context_scale,**arg_null)[0]

                diff_uncond_text  = noise_pred_cond - noise_pred_drop_text
                diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_text, 
                                                                                                    noise_pred_cond, 
                                                                                                    momentum_buffer=text_momentumbuffer, 
                                                                                                    norm_threshold=55) \
                        + (audio_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_audio, 
                                                                                noise_pred_cond, 
                                                                                momentum_buffer=audio_momentumbuffer, 
                                                                                norm_threshold=55)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = temp_x0.squeeze(0)
                
                if t == timesteps[-1]:
                    if strength == 0:
                        latents = torch.where(unity_masks == 1.0, unity_latents, latents)
                    if identity_strength == 0:
                        latents = torch.where(unity_masks == 0.0, unity_latents, latents)
                    

            x0 = [latents]
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None