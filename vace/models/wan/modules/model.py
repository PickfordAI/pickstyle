# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.amp as amp
import torch.nn as nn
import numpy as np
from diffusers.configuration_utils import register_to_config
try:
    from vace.wan.modules.model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d
except ModuleNotFoundError:
    from wan.modules.model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d


class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, e, seq_lens, grid_sizes, freqs, context, context_lens)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
            
        return x
       
class VaceWanModel(WanModel):
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        
        self.enable_teacache = False
        
        
    def change_lora_state(self, adapter_name, activate):
        updated = False
        for name, module in self.vace_blocks.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                module.scaling[adapter_name] = self.lora_scaling if activate else 0.0
                updated = True
        assert updated, "No VACE blocks found with LoRA parameters."


    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        use_gradient_checkpointing,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.vace_blocks:
            if use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, new_kwargs['x'], new_kwargs['e'], new_kwargs['seq_lens'], new_kwargs['grid_sizes'], self.freqs, 
                        new_kwargs['context'], new_kwargs['context_lens'],
                        use_reentrant=False
                    )
            else: 
                c = block(c, **new_kwargs)
                
        hints = torch.unbind(c)[:-1]
        return hints
    
    def activate_teacache(self):
        self.enable_teacache = True
        self.__class__.cnt = 0
    
    def deactivate_teacache(self):
        self.enable_teacache = False
        self.__class__.cnt = 0
    
    def teacache_init(
        self,
        use_ret_steps=True,
        teacache_thresh=0.2,
        sample_steps=40,
        ckpt_dir='',
    ):
        self.enable_teacache = True
        
        self.__class__.cnt = 0
        self.__class__.num_steps = sample_steps*3
        self.__class__.teacache_thresh = teacache_thresh
        self.__class__.accumulated_rel_l1_distance_even = 0
        self.__class__.accumulated_rel_l1_distance_odd = 0
        self.__class__.previous_e0_even = None
        self.__class__.previous_e0_odd = None
        self.__class__.previous_residual_even = None
        self.__class__.previous_residual_odd = None
        self.__class__.use_ret_steps = use_ret_steps

        if use_ret_steps:
            if '1.3B' in ckpt_dir:
                self.__class__.coefficients = [ 2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
            if '14B' in  ckpt_dir:
                self.__class__.coefficients = [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]
            self.__class__.ret_steps = 5*3
            self.__class__.cutoff_steps = sample_steps*3
        else:
            if '1.3B' in ckpt_dir:
                self.__class__.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
        
            if '14B' in ckpt_dir:
                self.__class__.coefficients = [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
            self.__class__.ret_steps = 1*3
            self.__class__.cutoff_steps = sample_steps*3 - 3
        print("[INFO] teacache_init done")

    def forward_backbone(self, x, vace_context, vace_context_scale, seq_len, use_gradient_checkpointing, use_gradient_checkpointing_offload, kwargs):
        hints = self.forward_vace(x, vace_context, seq_len, use_gradient_checkpointing, kwargs)
        
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale
        
        def create_custom_forward(module):
            def custom_forward(x, hints, context_scale, e, seq_lens, grid_sizes, freqs, context, context_lens):
                return module(x, hints, context_scale=context_scale, e=e, seq_lens=seq_lens,
                        grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=context_lens)
            return custom_forward
        
        for block in self.blocks:
            if use_gradient_checkpointing:
                e0 = kwargs['e']
                seq_lens = kwargs['seq_lens']
                grid_sizes = kwargs['grid_sizes']
                context = kwargs['context']
                context_lens = kwargs['context_lens']
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, hints, vace_context_scale, e0, seq_lens, grid_sizes, self.freqs, context, context_lens,
                            use_reentrant=False
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, hints, vace_context_scale, e0, seq_lens, grid_sizes, self.freqs, context, context_lens,
                            use_reentrant=False
                        )
            else:
                x = block(x, **kwargs)
                
        return x
    
    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dtype in [torch.float32, torch.float64, torch.long]:
            with amp.autocast('cuda', dtype=torch.float32):
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).to(t.dtype))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)
        
        # teacache
        if self.enable_teacache:
            modulated_inp = e0 if self.use_ret_steps else e
            if self.cnt%3==0: # cond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_cond = True
                    self.accumulated_rel_l1_distance_cond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_cond += rescale_func(((modulated_inp-self.previous_e0_cond).abs().mean() / self.previous_e0_cond.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_cond < self.teacache_thresh:
                        should_calc_cond = False
                    else:
                        should_calc_cond = True
                        self.accumulated_rel_l1_distance_cond = 0
                self.previous_e0_cond = modulated_inp.clone()
            elif self.cnt%3==1: # drop_text
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_drop_text = True
                    self.accumulated_rel_l1_distance_drop_text = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_drop_text += rescale_func(((modulated_inp-self.previous_e0_drop_text).abs().mean() / self.previous_e0_drop_text.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_drop_text < self.teacache_thresh:
                        should_calc_drop_text = False
                    else:
                        should_calc_drop_text = True
                        self.accumulated_rel_l1_distance_drop_text = 0
                self.previous_e0_drop_text = modulated_inp.clone()
            else: # uncond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_uncond = True
                    self.accumulated_rel_l1_distance_uncond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_uncond += rescale_func(((modulated_inp-self.previous_e0_uncond).abs().mean() / self.previous_e0_uncond.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_uncond < self.teacache_thresh:
                        should_calc_uncond = False
                    else:
                        should_calc_uncond = True
                        self.accumulated_rel_l1_distance_uncond = 0
                self.previous_e0_uncond = modulated_inp.clone()
                

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        if self.enable_teacache:
            if self.cnt%3==0:
                if not should_calc_cond:
                    x +=  self.previous_residual_cond
                else:
                    ori_x = x.clone()
                    x = self.forward_backbone(x, vace_context, vace_context_scale, seq_len, use_gradient_checkpointing, use_gradient_checkpointing_offload, kwargs)
                    self.previous_residual_cond = x - ori_x
            elif self.cnt%3==1:
                if not should_calc_drop_text:
                    x +=  self.previous_residual_drop_text
                else:
                    ori_x = x.clone()
                    x = self.forward_backbone(x, vace_context, vace_context_scale, seq_len, use_gradient_checkpointing, use_gradient_checkpointing_offload, kwargs)
                    self.previous_residual_drop_text = x - ori_x
            else:
                if not should_calc_uncond:
                    x +=  self.previous_residual_uncond
                else:
                    ori_x = x.clone()
                    x = self.forward_backbone(x, vace_context, vace_context_scale, seq_len, use_gradient_checkpointing, use_gradient_checkpointing_offload, kwargs)
                    self.previous_residual_uncond = x - ori_x
        else:
            x = self.forward_backbone(x, vace_context, vace_context_scale, seq_len, use_gradient_checkpointing, use_gradient_checkpointing_offload, kwargs)
        
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        
        if self.enable_teacache:
            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0
                
        return [u.float() for u in x]