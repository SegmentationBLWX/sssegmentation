'''
Function:
    Scripts for removing optimizer from checkpoints
Author:
    Zhenchao Jin
'''
import os
import glob
import torch
from tqdm import tqdm


'''remove'''
def remove(directory):
    filepaths = glob.glob(os.path.join(directory, '*.pth'))
    pbar = tqdm(filepaths)
    for filepath in pbar:
        pbar.set_description(filepath)
        ckpts = torch.load(filepath, map_location='cpu')
        if 'optimizer' in ckpts:
            del ckpts['optimizer']
        torch.save(ckpts, filepath)
    print('All done.')


'''DEBUG'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Scripts for removing optimizer from checkpoints.')
    parser.add_argument('--dir', dest='dir', help='directory which contains require-processing checkpoints', required=True, type=str)
    args = parser.parse_args()
    remove(args.dir)