'''
Function:
    setup sssegmentation
Author:
    Zhenchao Jin
'''
import ssseg
from setuptools import setup, find_packages


'''readme'''
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


'''setup'''
setup(
    name=ssseg.__title__,
    version=ssseg.__version__,
    description=ssseg.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent'
    ],
    author=ssseg.__author__,
    url=ssseg.__url__,
    author_email=ssseg.__email__,
    license=ssseg.__license__,
    include_package_data=True,
    install_requires=['torch', 'torchvision', 'mmcv-full'],
    zip_safe=True,
    packages=find_packages()
)