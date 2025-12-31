import json
import os
import imageio
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import av
import random


class VideoAug:
    def __init__(self):
        pass
    
    def identity(self, repeated_frames, **kwargs):
        """
        No augmentation, just return the repeated frames as is.
        """
        return repeated_frames
    
    def zoom_in(self, repeated_frames, zoom_factor=1.5, **kwargs):
        C, T, H, W = repeated_frames.shape
        output = []
        for t in range(T):
            scale = zoom_factor ** (t / (T - 1))  # increases over time
            h_crop, w_crop = int(H / scale), int(W / scale)
            # crop center
            top = (H - h_crop) // 2
            left = (W - w_crop) // 2
            frame = repeated_frames[:, t, top:top + h_crop, left:left + w_crop]
            frame = F.interpolate(frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            output.append(frame)
        return torch.cat(output, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
    
    def zoom_out(self, repeated_frames, zoom_factor=1.5, **kwargs):  # zoom_factor > 1.0 for zoom-in start
        C, T, H, W = repeated_frames.shape
        output = []
        for t in range(T):
            # Inverse schedule: scale decreases from zoom_factor to 1.0
            scale = zoom_factor ** (1 - t / (T - 1))
            h_crop, w_crop = int(H / scale), int(W / scale)
            # Crop central region
            top = (H - h_crop) // 2
            left = (W - w_crop) // 2
            frame = repeated_frames[:, t, top:top + h_crop, left:left + w_crop]
            # Upsample cropped region back to original size
            frame = F.interpolate(frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            output.append(frame)
        return torch.cat(output, dim=0).permute(1, 0, 2, 3)
    
    def slide_right(self, repeated_frames, zoom_factor=1.2, max_shift=200, **kwargs):
        C, T, H, W = repeated_frames.shape
        output = []
        for t in range(T):
            # Step 1: Zoom parameters
            scale = zoom_factor
            h_crop, w_crop = int(H / scale), int(W / scale)
            top = (H - h_crop) // 2

            # Step 2: Move crop window rightward → simulates leftward motion
            shift = int((max_shift * (t / (T - 1))**2))
            left = min(shift, W - w_crop)  # ensure crop stays within bounds

            frame = repeated_frames[:, t, top:top + h_crop, left:left + w_crop]

            # Step 3: Resize back to original resolution
            frame = F.interpolate(frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            output.append(frame)

        return torch.cat(output, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
    
    def slide_left(self, repeated_frames, zoom_factor=1.2, max_shift=200, **kwargs):
        C, T, H, W = repeated_frames.shape
        output = []
        for t in range(T):
            # Step 1: Zoom parameters
            scale = zoom_factor
            h_crop, w_crop = int(H / scale), int(W / scale)
            top = (H - h_crop) // 2

            # Step 2: Move crop window leftward → simulates rightward motion
            shift = int((max_shift * (t / (T - 1))**2))
            left = max(W - w_crop - shift, 0)  # move crop leftward

            frame = repeated_frames[:, t, top:top + h_crop, left:left + w_crop]

            # Step 3: Resize back to original resolution
            frame = F.interpolate(frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            output.append(frame)

        return torch.cat(output, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
    
    def slide_up(self, repeated_frames, zoom_factor=1.2, max_shift=200, **kwargs):
        C, T, H, W = repeated_frames.shape
        output = []
        for t in range(T):
            # Step 1: Zoomed-in crop from top-right (initial frame aligned with bottom)
            scale = zoom_factor
            h_crop, w_crop = int(H / scale), int(W / scale)
            left = W - w_crop  # always right-aligned

            # Instead of fixed top, we shift the crop window upward over time
            shift = int(max_shift * (t / (T - 1))**2)  # increases over time
            top = max(H - h_crop - shift, 0)  # start from bottom and move up

            frame = repeated_frames[:, t, top:top + h_crop, left:left + w_crop]

            # Step 2: Resize to original resolution
            frame = F.interpolate(frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            output.append(frame)
        
        return torch.cat(output, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
    
    def slide_down(self, repeated_frames, zoom_factor=1.2, max_shift=200, **kwargs):
        C, T, H, W = repeated_frames.shape
        output = []
        for t in range(T):
            # Step 1: Zoom parameters
            scale = zoom_factor
            h_crop, w_crop = int(H / scale), int(W / scale)
            left = W - w_crop  # right-aligned

            # Step 2: Slide downward by increasing top offset
            shift = int(max_shift * (t / (T - 1))**2)
            top = min(shift, H - h_crop)  # ensure crop stays within bounds

            frame = repeated_frames[:, t, top:top + h_crop, left:left + w_crop]

            # Step 3: Resize to original resolution
            frame = F.interpolate(frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            output.append(frame)

        return torch.cat(output, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
    
    def rotate(self, repeated_frames, max_angle=30, **kwargs):
        """
        Rotate the image smoothly over time, then zoom in to avoid black edges.
        max_angle: maximum rotation angle in degrees (can be positive or negative)
        """
        C, T, H, W = repeated_frames.shape
        zoom_factor = 1.5
        output = []
        
        for t in range(T):
            # Linear progression from 0 to max_angle
            progress = t / max(T - 1, 1)
            angle = max_angle * progress
            
            frame = repeated_frames[:, t]  # [C, H, W]
            
            # Step 1: Apply rotation first
            frame_rotated = TF.rotate(frame, angle, fill=0)
            
            # Step 2: Zoom in to crop out black edges after rotation
            h_crop, w_crop = int(H / zoom_factor), int(W / zoom_factor)
            top = (H - h_crop) // 2
            left = (W - w_crop) // 2
            frame_cropped = frame_rotated[:, top:top + h_crop, left:left + w_crop]
            frame_final = F.interpolate(frame_cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            
            output.append(frame_final.unsqueeze(0))
        
        return torch.cat(output, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)


class VideoStyleDataset(Dataset):
    def __init__(
        self, style, styles_list, basedir, video_resolution=(81, 480, 832), version=1
    ):
        assert style in styles_list, f"Style must be one of: {styles_list}"
        assert version in [1, 2], "Version must be either 1 or 2"
        if version == 1: # Unity and Style pairs
            self.style_paths = sorted(glob(os.path.join(basedir, style, "*.png")))
            self.src_paths = sorted(glob(os.path.join(basedir, 'unity', "*.png")))
            with open(os.path.join(basedir, 'caption.json'), 'r') as f:
                self.annotations = json.load(f)
        else: # Omniconsistency pairs
            self.style_paths = sorted(glob(os.path.join(basedir, style, "tar", "*.png")))
            self.src_paths = sorted(glob(os.path.join(basedir, style, "src", "*.png")))
            self.annotations = sorted(glob(os.path.join(basedir, style, "caption", "*.txt")))
            style = style.lower().replace("_", " ")
            if style == "clay toy":
                style = "clay"
        assert len(self.style_paths) == len(self.src_paths), f"Style and source paths must be the same length. Found {len(self.style_paths)} style and {len(self.src_paths)} source on style {style} version={version}."
            
        self.style = style
        self.version = version
        self.video_resolution = video_resolution
        self.T, self.H, self.W = video_resolution
        self.video_aug = VideoAug()
            
    def __len__(self):
        return len(self.style_paths)
    
    def preprocess_frame(self, frame_np: np.ndarray) -> torch.Tensor:
        """
        1) frame_np is HxWxC in [0..255]
        2) convert to [1, C, H, W]
        3) run VACE's `resize_crop` logic to get [1, C, self.H, self.W]
        4) normalize to [-1, +1]
        5) return [C, self.H, self.W]
        """
        frame = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float()  
        # now frame.shape = [1, C, H_orig, W_orig]
        ih, iw = frame.shape[2], frame.shape[3]
        if ih != self.H or iw != self.W:
            scale = max(self.W / iw, self.H / ih)
            new_h = int(round(scale * ih))
            new_w = int(round(scale * iw))
            frame = F.interpolate(frame, size=(new_h, new_w), mode="bicubic", antialias=True)
            y1 = (new_h - self.H) // 2
            x1 = (new_w - self.W) // 2
            frame = frame[:, :, y1 : y1 + self.H, x1 : x1 + self.W]
        # normalize to [-1, +1]
        frame = frame.div(127.5).sub_(1.0)
        return frame.squeeze(0)  # [C, H, W]


    def load_random_repeated_frames(self, img_path):
        frame_np = imageio.v2.imread(img_path)[..., :3] # Ensure it's RGB (Not RGBA)
        one_frame = self.preprocess_frame(frame_np)
        if self.T > 1:
            repeated_frames = one_frame.unsqueeze(1).repeat(1, self.T, 1, 1)
        else:
            repeated_frames = one_frame.unsqueeze(1)
        return repeated_frames
    

    def save_frame(self, frame_tensor, filename):
        from PIL import Image
        img = frame_tensor.clone().cpu()
        # Undo VACE normalization: [-1,+1] -> [0,255]
        img = (img + 1.0) * 127.5
        img = img.clamp(0, 255).to(torch.uint8)
        np_img = img.permute(1, 2, 0).numpy() 
        Image.fromarray(np_img).save(filename)
    

    def __getitem__(self, idx):
        style_path = self.style_paths[idx]
        src_path = self.src_paths[idx]
        
        if self.version == 1:
            video_name = os.path.basename(src_path)
            description = self.annotations.get(video_name, "")
            if description != "":
                description = " " + description
            caption = f"{self.style} style.{description}"
        elif self.version == 2:
            annotation_path = self.annotations[idx]
            with open(annotation_path, 'r') as f:
                caption = f.read().strip()
        else:
            raise ValueError("Version must be either 1 or 2")
        
        video_src = self.load_random_repeated_frames(src_path)
        video_style = self.load_random_repeated_frames(style_path)
        mask_src = torch.ones_like(video_src)
        
        aug_methods = [self.video_aug.zoom_in, self.video_aug.zoom_out, self.video_aug.slide_left, self.video_aug.slide_right, self.video_aug.slide_up, self.video_aug.slide_down, self.video_aug.rotate, self.video_aug.rotate]
        if random.random() < 0.8 and self.T > 1:
            selected_aug = random.choice(aug_methods)
        else:
            selected_aug = self.video_aug.identity
        aug_kwargs = {
            'zoom_factor': random.uniform(1.2, 2), 
            'max_shift': random.randint(100, 200),
            'max_angle': random.uniform(-20, 20)
        }
        
        return selected_aug(video_src, **aug_kwargs), mask_src, selected_aug(video_style, **aug_kwargs), caption
    
    
    
class VideoUnityDataset(Dataset):
    def __init__(self, basedir, video_resolution=(81, 480, 832), use_caption=False):
        self.video_dir = os.path.join(basedir, 'unity_vid')
        self.video_paths = sorted(glob(os.path.join(self.video_dir, "*.mp4")))
        self.video_resolution = video_resolution
        self.T, self.H, self.W = video_resolution
        self.use_caption = use_caption
        self.caption_file = os.path.join(basedir, 'caption.json')
        if use_caption and os.path.exists(self.caption_file):
            with open(self.caption_file, 'r') as f:
                self.captions_map = json.load(f)
        else:
            self.captions_map = {}

    def __len__(self):
        # return len(self.video_paths)
        return 5

    def preprocess_frame(self, frame_np: np.ndarray) -> torch.Tensor:
        frame = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float()
        ih, iw = frame.shape[2], frame.shape[3]
        if ih != self.H or iw != self.W:
            scale = max(self.W / iw, self.H / ih)
            new_h = int(round(scale * ih))
            new_w = int(round(scale * iw))
            frame = F.interpolate(frame, size=(new_h, new_w), mode="bicubic", antialias=True)
            y1 = (new_h - self.H) // 2
            x1 = (new_w - self.W) // 2
            frame = frame[:, :, y1: y1 + self.H, x1: x1 + self.W]
        frame = frame.div(127.5).sub_(1.0)
        return frame.squeeze(0)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path)
        frames = []
        try:
            container = av.open(video_path)
            for i, frame in enumerate(container.decode(video=0)):
                if i >= self.T:
                    break
                img = frame.to_rgb().to_ndarray()
                frame_tensor = self.preprocess_frame(img)
                frames.append(frame_tensor)
            container.close()
        except Exception as e:
            print(f"Error reading {video_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.video_paths))

        # Pad or truncate to T
        if len(frames) < self.T:
            pad = self.T - len(frames)
            frames.extend([frames[-1]] * pad)
        video_src = torch.stack(frames, dim=1)  # [C, T, H, W]

        mask_src = torch.ones_like(video_src)
        caption = self.captions_map.get(video_name, "")
        caption = f"reconstruct. {caption}" if caption else "reconstruct."
        return video_src, mask_src, video_src.clone(), caption
    
    

def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


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
    
    
def video_collate_fn(batch):
    videos_src, mask_src, videos_style, captions = zip(*batch)
    if len(videos_src) == 1:
        # If only one video, no need to stack (saves time)
        videos_src = videos_src[0].unsqueeze(0)
        masks_src = mask_src[0].unsqueeze(0)
        videos_style = videos_style[0].unsqueeze(0)
    else:
        videos_src = torch.stack(videos_src) 
        masks_src = torch.stack(mask_src)
        videos_style = torch.stack(videos_style)
    return videos_src, masks_src, videos_style, list(captions)

   
def _test_vid():
    video_dataset = VideoStyleDataset(
        style='pixar',
        styles_list=['pixar'],        
        basedir='dataset_style',
    )
    rand_idx = random.randint(0, len(video_dataset) - 1)    
    video_src, mask_src, video_dst, caption = video_dataset[rand_idx]
    print(f"Video shape: {video_src.shape}, Mask shape: {mask_src.shape}, Video style shape: {video_dst.shape} Caption: {caption}")
    cache_video(
        tensor=torch.stack([video_src, video_dst], dim=0),
        save_file='dataset_style_test.mp4',
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )


if __name__ == "__main__":
    import binascii
    import os
    import torchvision
    
    import os.path as osp
    _test_vid()
