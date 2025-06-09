'''
Function:
    Implementation of IO related operations
Author:
    Zhenchao Jin
'''
import os
import torch
import tempfile
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm


'''downloadwithprogress'''
def downloadwithprogress(url, filename):
    import requests
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(desc=f"Downloading {os.path.basename(filename)}", total=total, unit='B', unit_scale=True, unit_divisor=1024) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


'''judgefileexist'''
def judgefileexist(filepath):
    if os.path.islink(filepath):
        filepath = os.readlink(filepath)
    return os.path.exists(filepath)


'''touchdir'''
def touchdir(directory):
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except:
            pass


'''touchdirs'''
def touchdirs(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except:
            pass


'''loadckpts'''
def loadckpts(ckptspath, map_to_cpu=True):
    if os.path.islink(ckptspath):
        ckptspath = os.readlink(ckptspath)
    if ckptspath.startswith("http://") or ckptspath.startswith("https://"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        downloadwithprogress(ckptspath, tmp_path)
        ckpts = torch.load(tmp_path, map_location='cpu' if map_to_cpu else None)
        os.remove(tmp_path)
    else:
        ckpts = torch.load(ckptspath, map_location='cpu' if map_to_cpu else None)
    return ckpts


'''saveckpts'''
def saveckpts(ckpts, savepath, make_soft_link=True, soft_link_dst=None):
    save_response = torch.save(ckpts, savepath)
    if make_soft_link:
        if soft_link_dst is None:
            soft_link_dst = os.path.join(os.path.dirname(savepath), 'checkpoints-epoch-latest.pth')
        symlink(savepath, soft_link_dst)
    return save_response


'''symlink'''
def symlink(src_path, dst_path):
    if os.path.islink(dst_path):
        os.unlink(dst_path)
    os.symlink(src_path, dst_path)
    return True


'''loadpretrainedweights'''
def loadpretrainedweights(structure_type, pretrained_model_path='', default_model_urls={}, map_to_cpu=True, possible_model_keys=['model', 'state_dict']):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location='cpu') if map_to_cpu else torch.load(pretrained_model_path)
    else:
        checkpoint = model_zoo.load_url(default_model_urls[structure_type], map_location='cpu') if map_to_cpu else model_zoo.load_url(default_model_urls[structure_type])
    state_dict = checkpoint
    for key in possible_model_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    return state_dict