'''
Function:
    Load the VSPW dataset
Author:
    Zhenchao Jin
'''
import os
import random
from .base import BaseDataset


'''VSPWDataset'''
class VSPWDataset(BaseDataset):
    num_classes = 124
    classnames = [
        'wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator', 'Playground_slide', 'handrail_or_fence', 'window', 
        'rail', 'goal', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field', 'road', 'path', 'crosswalk', 
        'building', 'house', 'bridge', 'tower', 'windmill', 'well_or_well_lid', 'other_construction', 'sky', 'mountain', 'stone', 
        'wood', 'ice', 'snowfield', 'grandstand', 'sea', 'river', 'lake', 'waterfall', 'water', 'billboard_or_Bulletin_Board', 
        'sculpture', 'pipeline', 'flag', 'parasol_or_umbrella', 'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 
        'bicycle', 'motorcycle', 'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre', 'traffic_light', 'lamp', 'person', 
        'cat', 'dog', 'horse', 'cattle', 'other_animal', 'tree', 'flower', 'other_plant', 'toy', 'ball_net', 'backboard', 'skateboard', 
        'bat', 'ball', 'cupboard_or_showcase_or_storage_rack', 'box', 'traveling_case_or_trolley_case', 'basket', 'bag_or_package', 
        'trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk', 
        'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'other_machine', 'refrigerator', 'washing_machine', 
        'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes', 'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book', 'tool', 
        'blackboard', 'tissue', 'screen_or_television', 'computer', 'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product', 'fruit', 
        'food', 'instrument', 'train'
    ]
    clsid2label = {0: 255, 254: 255}
    for i in range(1, num_classes+1): clsid2label[i] = i - 1
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(VSPWDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'data')
        self.ann_dir = os.path.join(rootdir, 'data')
        # obatin imageids
        self.imageids, self.annids = [], []
        with open(os.path.join(rootdir, dataset_cfg['set']+'.txt')) as fp:
            dirnames = fp.readlines()
            for dirname in dirnames:
                dirname = dirname.strip()
                if not dirname: continue
                if mode == 'TRAIN':
                    self.imageids.append(dirname)
                else:
                    for imgname in os.listdir(os.path.join(self.image_dir, dirname, 'origin')):
                        imageid = f'{dirname}/origin/{imgname}'
                        annid = f'{dirname}/mask/{imgname.replace(".jpg", ".png")}'
                        self.imageids.append(imageid)
                        self.annids.append(annid)
    '''pull item'''
    def __getitem__(self, index):
        if self.mode == 'TRAIN':
            imagedir = os.path.join(self.image_dir, self.imageids[index], 'origin')
            imgname = random.choice(os.listdir(imagedir))
            imagepath = os.path.join(imagedir, imgname)
            annpath = os.path.join(self.ann_dir, self.imageids[index], f'mask/{imgname.replace(".jpg", ".png")}')
        else:
            imagepath = os.path.join(self.image_dir, self.imageids[index])
            annpath = os.path.join(self.ann_dir, self.annids[index])
        if self.dataset_cfg['set'] == 'test': self.dataset_cfg['with_ann'] = False
        sample = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', True))
        sample.update({'id': self.imageids[index]})
        if self.mode == 'TRAIN':
            sample = self.synctransform(sample, 'without_totensor_normalize_pad')
            sample['edge'] = self.generateedge(sample['segmentation'].copy())
            sample = self.synctransform(sample, 'only_totensor_normalize_pad')
        else:
            sample = self.synctransform(sample, 'all')
        return sample
    '''length'''
    def __len__(self):
        return len(self.imageids)