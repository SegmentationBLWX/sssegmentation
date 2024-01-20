'''
Function:
    Scripts for collecting develop environment
Author:
    Zhenchao Jin
'''
from ssseg.modules import EnvironmentCollector


'''run'''
if __name__ == '__main__':
    for key, value in EnvironmentCollector().collectenv().items():
        print(f'{key}: {value}')