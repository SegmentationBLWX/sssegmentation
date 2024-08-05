'''
Function:
    Implementation of misc
Author:
    Zhenchao Jin
'''
import os
import torch
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from threading import Thread


'''getsdpasettings'''
def getsdpasettings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn("Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.", category=UserWarning, stacklevel=2)
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).", 
                category=UserWarning, stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True
    return old_gpu, use_flash_attn, math_kernel_on


'''getconnectedcomponents'''
def getconnectedcomponents(mask):
    from ssseg import _C
    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


'''masktobox'''
def masktobox(masks):
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing='xy')
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
    return bbox_coords


'''loadimgastensor'''
def loadimgastensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size
    return img, video_height, video_width


'''AsyncVideoFrameLoader'''
class AsyncVideoFrameLoader:
    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self._images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        # load the first frame to fill video_height and video_width and also to cache it (since it's most likely where the user will click)
        self.__getitem__(0)
        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e
        # set thread
        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()
    '''getitem'''
    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception
        img = self.images[index]
        if img is not None:
            return img
        img, video_height, video_width = loadimgastensor(self.img_paths[index], self.image_size)
        self.video_height = video_height
        self.video_width = video_width
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.cuda(non_blocking=True)
        self.images[index] = img
        return img
    '''len'''
    def __len__(self):
        return len(self.images)


'''loadvideoframes'''
def loadvideoframes(video_path, image_size, offload_video_to_cpu, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225), async_loading_frames=False):
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError("Only JPEG frames are supported at this moment")
    frame_names = [
        p for p in os.listdir(jpg_folder) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(img_paths, image_size, offload_video_to_cpu, img_mean, img_std)
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = loadimgastensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


'''fillholesinmaskscores'''
def fillholesinmaskscores(mask, max_area):
    # Holes are those connected components in background with area <= self.max_area (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"
    labels, areas = getconnectedcomponents(mask <= 0)
    is_hole = (labels > 0) & (areas <= max_area)
    # We fill holes with a small positive mask score (0.1) to change them to foreground.
    mask = torch.where(is_hole, 0.1, mask)
    # return
    return mask


'''concatpoints'''
def concatpoints(old_point_inputs, new_points, new_labels):
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)
    return {"point_coords": points, "point_labels": labels}


'''selectclosestcondframes'''
def selectclosestcondframes(frame_idx, cond_frame_outputs, max_cond_frame_num):
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]
        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]
        # add other temporally closest conditioning frames until reaching a total of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted((t for t in cond_frame_outputs if t not in selected_outputs), key=lambda x: abs(x - frame_idx))[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}
    return selected_outputs, unselected_outputs


'''get1dsinepe'''
def get1dsinepe(pos_inds, dim, temperature=10000):
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


'''inittxy'''
def inittxy(end_x, end_y):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


'''computeaxialcis'''
def computeaxialcis(dim, end_x, end_y, theta=10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = inittxy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


'''reshapeforbroadcast'''
def reshapeforbroadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


'''applyrotaryenc'''
def applyrotaryenc(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshapeforbroadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)