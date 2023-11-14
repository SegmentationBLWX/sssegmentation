'''
Function:
    Implementation of IO related operations
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.utils.model_zoo as model_zoo


'''judgefileexist'''
def judgefileexist(filepath):
    if os.path.islink(filepath):
        filepath = os.readlink(filepath)
    return os.path.exists(filepath)


'''touchdir'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        try: os.mkdir(dirname)
        except: pass
        return False
    return True


'''loadckpts'''
def loadckpts(ckptspath, map_to_cpu=True):
    if os.path.islink(ckptspath):
        ckptspath = os.readlink(ckptspath)
    if map_to_cpu: 
        ckpts = torch.load(ckptspath, map_location=torch.device('cpu'))
    else: 
        ckpts = torch.load(ckptspath)
    return ckpts


'''saveckpts'''
def saveckpts(ckpts, savepath, make_soft_link=True, soft_link_dst=None):
    save_response = torch.save(ckpts, savepath)
    if make_soft_link:
        if soft_link_dst is None:
            soft_link_dst = os.path.join(os.path.dirname(savepath), 'epoch_last.pth')
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