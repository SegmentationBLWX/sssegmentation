'''
Function:
    Implementation of CityScapesDataset
Author:
    Zhenchao Jin
'''
import os
import numpy as np
import pandas as pd
from PIL import Image
from .base import BaseDataset


'''CityScapesDataset'''
class CityScapesDataset(BaseDataset):
    num_classes = 19
    classnames = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 
        'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    palette = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
        (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    clsid2label = {
        -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
        7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 
        15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 
        31: 16, 32: 17, 33: 18
    }
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(CityScapesDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'leftImg8bit', dataset_cfg['set'])
        self.ann_dir = os.path.join(rootdir, 'gtFine', dataset_cfg['set'])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
        self.ann_ext = '.png'
        self.image_ext = '.png'
    '''getitem'''
    def __getitem__(self, index):
        # imageid
        imageid = self.imageids[index % len(self.imageids)]
        # read sample_meta
        imagepath = os.path.join(self.image_dir, f'{imageid}{self.image_ext}')
        annpath = os.path.join(self.ann_dir, f'{imageid.replace("leftImg8bit", "gtFine_labelIds")}{self.ann_ext}')
        sample_meta = self.read(imagepath, annpath)
        # add image id
        sample_meta.update({'id': imageid})
        # synctransforms
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta
    '''format results for test set of Cityscapes'''
    @staticmethod
    def formatresults(results, filenames, to_label_id=True, savedir='results'):
        assert len(filenames) == len(results)
        def convert(result):
            import cityscapesscripts.helpers.labels as CSLabels
            result_copy = result.copy()
            for trainId, label in CSLabels.trainId2label.items():
                result_copy[result == trainId] = label.id
            return result_copy
        if not os.path.exists(savedir): os.mkdir(savedir)
        result_files = []
        for idx in range(len(results)):
            result = results[idx]
            filename = filenames[idx]
            if to_label_id: result = convert(result)
            basename = os.path.splitext(os.path.basename(filename))[0]
            png_filename = os.path.join(savedir, f'{basename}.png')
            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color
            output.putpalette(palette)
            output.save(png_filename)