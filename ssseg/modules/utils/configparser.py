'''
Function:
    Implementation of ConfigParser
Author:
    Zhenchao
'''
import os
import sys
import importlib
import dill as pickle
from .io import touchdir


'''ConfigParser'''
class ConfigParser():
    def __init__(self, library_root='ssseg'):
        self.library_root = self.preparelibraryroot(library_root)
    '''preparelibraryroot'''
    def preparelibraryroot(self, library_root):
        if not library_root.startswith('/'):
            library_root = '/' + library_root
        if not library_root.endswith('/'):
            library_root = library_root + '/'
        return library_root
    '''parsefrompy'''
    def parsefrompy(self, cfg_file_path):
        # assert
        assert cfg_file_path.endswith('.py')
        assert self.library_root in cfg_file_path, f'cfg_file_path should contain {self.library_root}'
        # obtain module path
        module_path = cfg_file_path[len(os.getcwd()):].replace('\\', '/')
        module_path = module_path.replace('/', '.')
        module_path = module_path.strip('.')[:-3]
        try:
            cfg = importlib.import_module(module_path, __package__)
        except:
            sys.path.insert(0, '.')
            cfg = importlib.import_module(module_path, __package__)
        # return cfg
        return cfg, cfg_file_path
    '''parsefrompkl'''
    def parsefrompkl(self, cfg_file_path):
        cfg = pickle.load(open(cfg_file_path, 'rb'))
        return cfg, cfg_file_path
    '''parse'''
    def parse(self, cfg_file_path):
        # ext to parse method
        ext_to_parse_method = {
            '.py': self.parsefrompy, '.pkl': self.parsefrompkl,
        }
        # config ext
        cfg_ext = os.path.splitext(cfg_file_path)[-1]
        # assert
        assert cfg_ext in ext_to_parse_method, f'unable to parse config with extension {cfg_ext}'
        # parse
        return ext_to_parse_method[cfg_ext](cfg_file_path=cfg_file_path)
    '''save'''
    def save(self, work_dir=''):
        work_dir = os.path.join(work_dir, 'configs')
        touchdir(work_dir)
        savepath = os.path.join(work_dir, os.path.basename(self.cfg_file_path) + '.pkl')
        return pickle.dump(self.cfg, open(savepath, 'wb'))
    '''call'''
    def __call__(self, cfg_file_path):
        cfg_file_path = os.path.abspath(os.path.expanduser(cfg_file_path))
        self.cfg, self.cfg_file_path = self.parse(cfg_file_path=cfg_file_path)
        return self.cfg, self.cfg_file_path