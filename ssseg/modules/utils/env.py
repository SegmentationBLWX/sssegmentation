'''
Function:
    EnvironmentCollector
Author:
    Zhenchao Jin
'''
import os
import sys
import torch
import subprocess
import numpy as np
from collections import OrderedDict, defaultdict


'''EnvironmentCollector'''
class EnvironmentCollector():
    def __init__(self, filter_env_keys=None):
        self.filter_env_keys = filter_env_keys
    '''collectenv'''
    def collectenv(self):
        from distutils import errors
        # basic info
        env_info = OrderedDict()
        env_info['System Platform'] = sys.platform
        env_info['Python Version'] = sys.version.replace('\n', '')
        env_info['CUDA Available'] = torch.cuda.is_available()
        env_info['MUSA Available'] = self.ismusaavailable()
        env_info['Numpy Random Seed'] = np.random.get_state()[1][0]
        # cuda info
        if env_info['CUDA Available']:
            devices = defaultdict(list)
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
            for name, device_ids in devices.items():
                env_info['GPU ' + ','.join(device_ids)] = name
            CUDA_HOME = self.getcudahome()
            env_info['CUDA HOME'] = CUDA_HOME
            if CUDA_HOME is not None and os.path.isdir(env_info['CUDA HOME']):
                if CUDA_HOME == '/opt/rocm':
                    try:
                        nvcc = os.path.join(CUDA_HOME, 'hip/bin/hipcc')
                        nvcc = subprocess.check_output(f'"{nvcc}" --version', shell=True)
                        nvcc = nvcc.decode('utf-8').strip()
                        release = nvcc.rfind('HIP version:')
                        build = nvcc.rfind('')
                        nvcc = nvcc[release:build].strip()
                    except subprocess.SubprocessError:
                        nvcc = 'Not Available'
                else:
                    try:
                        nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                        nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                        nvcc = nvcc.decode('utf-8').strip()
                        release = nvcc.rfind('Cuda compilation tools')
                        build = nvcc.rfind('Build ')
                        nvcc = nvcc[release:build].strip()
                    except subprocess.SubprocessError:
                        nvcc = 'Not Available'
                env_info['NVCC'] = nvcc
        elif env_info['MUSA Available']:
            devices = defaultdict(list)
            for k in range(torch.musa.device_count()):
                devices[torch.musa.get_device_name(k)].append(str(k))
            for name, device_ids in devices.items():
                env_info['GPU ' + ','.join(device_ids)] = name
            MUSA_HOME = self.getmusahome()
            env_info['MUSA HOME'] = MUSA_HOME
            if MUSA_HOME is not None and os.path.isdir(MUSA_HOME):
                try:
                    mcc = os.path.join(MUSA_HOME, 'bin/mcc')
                    subprocess.check_output(f'"{mcc}" -v', shell=True)
                except subprocess.SubprocessError:
                    mcc = 'Not Available'
                env_info['MCC'] = mcc
        # pytorch
        env_info['PyTorch'] = torch.__version__
        env_info['PyTorch Compiling Details'] = self.getpytorchbuildconfig().strip('\n').strip(' ')
        # torchvision
        try:
            import torchvision
            env_info['TorchVision'] = torchvision.__version__
        except ModuleNotFoundError:
            pass
        # cv2
        try:
            import cv2
            env_info['OpenCV'] = cv2.__version__
        except ImportError:
            pass
        # c++ compiler
        try:
            import io
            import sysconfig
            cc = sysconfig.get_config_var('CC')
            if cc:
                cc = os.path.basename(cc.split()[0])
                cc_info = subprocess.check_output(f'{cc} --version', shell=True)
                env_info['GCC'] = cc_info.decode('utf-8').partition('\n')[0].strip()
            else:
                import locale
                from distutils.ccompiler import new_compiler
                ccompiler = new_compiler()
                ccompiler.initialize()
                cc = subprocess.check_output(f'{ccompiler.cc}', stderr=subprocess.STDOUT, shell=True)
                encoding = os.device_encoding(sys.stdout.fileno()) or locale.getpreferredencoding()
                env_info['MSVC'] = cc.decode(encoding).partition('\n')[0].strip()
                env_info['GCC'] = 'n/a'
        except (subprocess.CalledProcessError, errors.DistutilsPlatformError):
            env_info['GCC'] = 'n/a'
        except io.UnsupportedOperation as e:
            env_info['MSVC'] = f'n/a, reason: {str(e)}'
        # filter
        if self.filter_env_keys is not None:
            for key in self.filter_env_keys:
                env_info.pop(key, None)
        # return
        return env_info
    '''ismusaavailable'''
    def ismusaavailable(self):
        try:
            import torch_musa
            return True
        except ImportError:
            return False
    '''getcudahome'''
    def getcudahome(self):
        if torch.__version__ == 'parrots':
            from parrots.utils.build_extension import CUDA_HOME
        else:
            if self.isrocmpytorch():
                from torch.utils.cpp_extension import ROCM_HOME
                CUDA_HOME = ROCM_HOME
            else:
                from torch.utils.cpp_extension import CUDA_HOME
        return CUDA_HOME
    '''getmusahome'''
    def getmusahome(self):
        return os.environ.get('MUSA_HOME')
    '''isrocmpytorch'''
    def isrocmpytorch(self):
        is_rocm = False
        if torch.__version__ != 'parrots':
            try:
                from torch.utils.cpp_extension import ROCM_HOME
                is_rocm = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
            except ImportError:
                pass
        return is_rocm
    '''getpytorchbuildconfig'''
    def getpytorchbuildconfig(self):
        if torch.__version__ == 'parrots':
            from parrots.config import get_build_info
            return get_build_info()
        else:
            return torch.__config__.show()