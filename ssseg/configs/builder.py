'''
Function:
    Implementation of BuildConfig
Author:
    Zhenchao Jin
'''
import os
import time
import shutil
import importlib
from distutils.dir_util import copy_tree


'''BuildConfig'''
def BuildConfig(cfg_file_path, tmp_cfg_dir='tmp_cfg'):
    # assert whether file exists
    assert os.path.exists(cfg_file_path), 'cfg_file_path %s not exist' % cfg_file_path
    # get config file info
    cfg_file_path = os.path.abspath(os.path.expanduser(cfg_file_path))
    cfg_info, ext = os.path.splitext(cfg_file_path)
    assert ext in ['.py'], 'only support .py type, but get %s' % ext
    cfg_dir, cfg_name = '/'.join(cfg_info.split('/')[:-1]), cfg_info.split('/')[-1]
    _base_cfg_dir = os.path.join('/'.join(cfg_info.split('/')[:-2]), '_base_')
    segmentor_dir = cfg_info.split('/')[-2]
    # base_cfg.py must exist
    base_cfg_file_path = os.path.join(cfg_dir, 'base_cfg' + ext)
    assert os.path.exists(base_cfg_file_path), 'base_cfg_file_path %s not exist' % base_cfg_file_path
    assert os.path.exists(_base_cfg_dir), '_base_cfg_dir %s not exist' % _base_cfg_dir
    # make temp dir for loading config
    if not os.path.exists(tmp_cfg_dir):
        try: os.mkdir(tmp_cfg_dir)
        except: pass
    if not os.path.exists(os.path.join(tmp_cfg_dir, segmentor_dir)):
        try: os.mkdir(os.path.join(tmp_cfg_dir, segmentor_dir))
        except: pass
    # copy config file and the base config file
    try: copy_tree(_base_cfg_dir, os.path.join(tmp_cfg_dir, '_base_'))
    except: pass
    shutil.copyfile(cfg_file_path, os.path.join(tmp_cfg_dir, segmentor_dir, cfg_name + ext))
    shutil.copyfile(base_cfg_file_path, os.path.join(tmp_cfg_dir, segmentor_dir, 'base_cfg' + ext))
    time.sleep(0.5)
    # load module from the temp dir
    try:
        cfg = importlib.import_module(f'{tmp_cfg_dir}.{segmentor_dir}.{cfg_name}', __package__)
    except:
        import sys
        sys.path.insert(0, '.')
        cfg = importlib.import_module(f'{tmp_cfg_dir}.{segmentor_dir}.{cfg_name}', __package__)
    # return cfg
    return cfg, cfg_file_path