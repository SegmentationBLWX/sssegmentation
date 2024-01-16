'''
Function:
    Implementation of COCOVOCSUBDataset, COCOStuff10kDataset and COCOStuffDataset
Author:
    Zhenchao Jin
'''
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from .base import BaseDataset


'''COCOVOCSUBDataset'''
class COCOVOCSUBDataset(BaseDataset):
    num_classes = 21
    classnames = [
        '__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorcycle', 'person', 
        'potted-plant', 'sheep', 'sofa', 'train', 'tv'
    ]
    palette = [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
        (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
        (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ]
    valid_clsids = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(COCOVOCSUBDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        from pycocotools import mask
        from pycocotools.coco import COCO
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, f"{dataset_cfg['set']}2017")
        # obatin imageids
        self.annfilepath = os.path.join(rootdir, f"annotations/instances_{dataset_cfg['set']}2017.json")
        self.coco_api = COCO(self.annfilepath)
        self.cocomask_api = mask
        self.imageids = []
        imageids_bar = tqdm(list(self.coco_api.imgs.keys()))
        for imageid in imageids_bar:
            imageids_bar.set_description('Preprocess imageid %s' % imageid)
            target = self.coco_api.loadAnns(self.coco_api.getAnnIds(imgIds=imageid))
            image_meta = self.coco_api.loadImgs(imageid)[0]
            seg_target = self.getsegtarget(target, image_meta['height'], image_meta['width'])
            if (seg_target > 0).sum() > 1000:
                self.imageids.append(imageid)
    '''getitem'''
    def __getitem__(self, index):
        # imageid
        imageid = self.imageids[index % len(self.imageids)]
        image_meta = self.coco_api.loadImgs(imageid)[0]
        imagepath = os.path.join(self.image_dir, image_meta['file_name'])
        # read image
        image = cv2.imread(imagepath)
        # read annotation
        seg_target = self.coco_api.loadAnns(self.coco_api.getAnnIds(imgIds=imageid))
        seg_target = self.getsegtarget(seg_target, image_meta['height'], image_meta['width'])
        # construct sample_meta
        sample_meta = {'image': image, 'seg_target': seg_target, 'width': image.shape[1], 'height': image.shape[0]}
        sample_meta.update({'id': str(imageid)})
        # synctransforms
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta
    '''getsegtarget'''
    def getsegtarget(self, target, height, width):
        segmentation = np.zeros((height, width), dtype=np.uint8)
        for instance in target:
            rle = self.cocomask_api.frPyObjects(instance['segmentation'], height, width)
            mask = self.cocomask_api.decode(rle)
            clsid = instance['category_id']
            if clsid not in self.valid_clsids: continue
            label = self.valid_clsids.index(clsid)
            if len(mask.shape) < 3: segmentation[:, :] += (segmentation == 0) * (mask * label)
            else: segmentation[:, :] += (segmentation == 0) * ((np.sum(mask, axis=2) > 0) * label).astype(np.uint8)
        return segmentation


'''COCOStuff10kDataset'''
class COCOStuff10kDataset(BaseDataset):
    num_classes = 182
    classnames = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other',
        'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff',
        'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other',
        'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
        'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
        'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea',
        'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table',
        'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'
    ]
    palette = BaseDataset.randompalette(num_classes)
    clsid2label = {0: 255}
    for i in range(1, num_classes+1): clsid2label[i] = i - 1
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(COCOStuff10kDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'images')
        self.ann_dir = os.path.join(rootdir, 'annotations')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, 'imageLists', dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
        self.ann_ext = '.mat'
        self.image_ext = '.jpg'


'''COCOStuffDataset'''
class COCOStuffDataset(BaseDataset):
    num_classes = 182
    classnames = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other',
        'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff',
        'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other',
        'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
        'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
        'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea',
        'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table',
        'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'
    ]
    palette = BaseDataset.randompalette(num_classes)
    clsid2label = {0: 255}
    for i in range(1, num_classes+1): clsid2label[i] = i - 1
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(COCOStuffDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        from pycocotools import mask
        from pycocotools.coco import COCO
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, f"{dataset_cfg['set']}2017")
        # obatin imageids
        self.annfilepath = os.path.join(rootdir, f"annotations/stuff_{dataset_cfg['set']}2017.json")
        self.coco_api = COCO(self.annfilepath)
        self.imageids = list(self.coco_api.imgs.keys())
    '''getitem'''
    def __getitem__(self, index):
        # imageid
        imageid = self.imageids[index % len(self.imageids)]
        # read sample_meta
        image_meta = self.coco_api.loadImgs(imageid)[0]
        imagepath = os.path.join(self.image_dir, image_meta['file_name'])
        annpath = imagepath.replace('jpg', 'png')
        sample_meta = self.read(imagepath, annpath)
        # add image id
        sample_meta.update({'id': str(imageid)})
        # synctransforms
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta