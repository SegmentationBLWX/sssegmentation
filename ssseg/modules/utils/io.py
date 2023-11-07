'''
Function:
    Implementation of IO related operations
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.utils.model_zoo as model_zoo


'''touchdir'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        try: os.mkdir(dirname)
        except: pass
        return False
    return True


'''loadckpts'''
def loadckpts(ckptspath, map_to_cpu=True):
    if map_to_cpu: 
        ckpts = torch.load(ckptspath, map_location=torch.device('cpu'))
    else: 
        ckpts = torch.load(ckptspath)
    return ckpts


'''saveckpts'''
def saveckpts(ckpts, savepath):
    save_response = torch.save(ckpts, savepath)
    return save_response


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