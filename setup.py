'''
Function:
    Setup sssegmentation
Author:
    Zhenchao Jin
'''
import ssseg
from setuptools import setup, find_packages


'''readme'''
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


'''parse the package dependencies listed in a requirements file but strips specific versioning information'''
def parserequirements(fname='requirements.txt', with_version=True):
    import re
    import sys
    from os.path import exists
    require_fpath = fname
    '''parse information from a line in a requirements text file'''
    def parseline(line):
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for info in parserequirefile(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]
                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info
    '''parse require file'''
    def parserequirefile(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parseline(line):
                        yield info
    '''gen packages items'''
    def genpackagesitems():
        if exists(require_fpath):
            for info in parserequirefile(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item
    # parse and return
    packages = list(genpackagesitems())
    return packages


'''setup'''
setup(
    name=ssseg.__title__,
    version=ssseg.__version__,
    description=ssseg.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent'
    ],
    author=ssseg.__author__,
    url=ssseg.__url__,
    author_email=ssseg.__email__,
    license=ssseg.__license__,
    include_package_data=True,
    install_requires=parserequirements('requirements.txt'),
    zip_safe=True,
    packages=find_packages()
)