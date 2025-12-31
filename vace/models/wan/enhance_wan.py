# models/wan/enhance_wan.py

import torch.nn.functional as F
from einops import rearrange
from wan.modules.model import WanSelfAttention
from enhance_a_video.globals import get_num_frames, set_num_frames, is_enhance_enabled
from enhance_a_video.enhance import enhance_score

def inject_enhance_for_vace(model):
    # only patch the `self_attn` attribute on each block
    for block in model.blocks:
        _patch_wan_self_attn(block.self_attn)
    # #vace_blocks with self_attn, patch them too:
    # if hasattr(model, 'vace_blocks'):
    #     for vblock in model.vace_blocks:
    #         _patch_wan_self_attn(vblock.self_attn)

def _patch_wan_self_attn(attn: WanSelfAttention):
    orig_forward = attn.forward

    def enhanced_forward(x, seq_lens, grid_sizes, freqs):
        # ==== compute q,k,v just as WanSelfAttention does internally ====
        b, s, n, d = *x.shape[:2], attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x)).view(b, s, n, d)
        k = attn.norm_k(attn.k(x)).view(b, s, n, d)

        # ==== build enhancement scores if enabled ====
        scores = None
        if is_enhance_enabled():
            # grid_sizes is [B,3] = (T_patches, H_patches, W_patches)
            T_p, H_p, W_p = grid_sizes[0].tolist()
            spatial = H_p * W_p

            q_img = rearrange(q, 'B (T S) N D -> (B S) N T D',
                              T=T_p, S=spatial)
            k_img = rearrange(k, 'B (T S) N D -> (B S) N T D',
                              T=T_p, S=spatial)

            scores = enhance_score(q_img, k_img, d, T_p)

        # ==== run the original attention ====
        out = orig_forward(x, seq_lens, grid_sizes, freqs)

        # ==== apply enhancement ====
        if scores is not None:
            out = out * scores

        return out

    attn.forward = enhanced_forward
