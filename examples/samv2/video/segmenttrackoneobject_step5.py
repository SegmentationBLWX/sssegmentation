'''
Function:
    SAMV2 examples: Segment & track one object
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Add a first click on a frame
# To get started, let's try to segment the child on the left.
# Here we make a positive click at (x, y) = (210, 350) with label `1`, by sending their coordinates and labels into the `addnewpoints` API.
# Note: label `1` indicates a positive click (to add a region) while label `0` indicates a negative click (to remove a region).
# the frame index we interact with
ann_frame_idx = 0
# give a unique id to each object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask, sending all clicks (and their labels) to `addnewpoints`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# run propagation throughout the video and collect the results in a dict (video_segments contains the per-frame segmentation results)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagateinvideo(inference_state):
    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
# further refine some details on this frame
ann_frame_idx = 150
# give a unique id to the object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
points = np.array([[82, 415]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([0], np.int32)
_, _, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# run propagation throughout the video and collect the results in a dict (video_segments contains the per-frame segmentation results)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagateinvideo(inference_state):
    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
# render the segmentation results every few frames
vis_frame_stride = 15
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        showmask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(f'out_frame_{out_frame_idx}.png')
    plt.cla()
    plt.clf()