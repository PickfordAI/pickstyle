# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp

import imageio
import torch
import torchvision

__all__ = ['cache_video', 'cache_image', 'str2bool']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name

def cache_single_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # tensor.shape: [B, C, T, H, W]
    # cache file
    cache_file = save_file or osp.join('/tmp', 'video_cache' + suffix)

    # ----- ensure shape is [T,C,H,W] -----
    if tensor.ndim == 5:   # [B,C,T,H,W] → pick first batch
        tensor = tensor[0]
    if tensor.ndim == 4 and tensor.shape[0] in (1,3):  
        # [C,T,H,W] → transpose to [T,C,H,W]
        tensor = tensor.permute(1,0,2,3)
    assert tensor.ndim == 4 and tensor.shape[1] in (1,3), f"Unexpected shape {tensor.shape}"

    # ----- normalization -----
    tensor = tensor.clamp(*value_range)
    if normalize:
        lo, hi = value_range
        tensor = (tensor - lo) / (hi - lo)

    # ----- scale & format -----
    frames = (tensor * 255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
    # shape: [T,H,W,C]

    # ----- write -----
    error = None
    for _ in range(retry):
        try:
            writer = imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=8)
            for f in frames:
                writer.append_data(f)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue

    print(f"cache_single_video failed, error: {error}")
    return None

def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ],
                                dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(
            cache_file, fps=fps, codec='libx264', quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
        return cache_file
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
