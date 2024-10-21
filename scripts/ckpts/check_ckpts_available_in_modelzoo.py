'''
Function:
    Check whether all checkpoints in ModelZoo is available
Author:
    Zhenchao Jin
'''
import os
import re
import pickle
import requests


'''isurlavailable'''
def isurlavailable(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False


'''checklinksinfile'''
def checklinksinfile(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    pattern = r'http[s]?://[^\s]+(?:\.pth|\.log)'
    links = re.findall(pattern, content)
    link_status = {}
    for link in links:
        is_accessible = isurlavailable(link)
        link_status[link] = 'Accessible' if is_accessible else 'Not Accessible'
        if not is_accessible:
            print(f'[Warning]: Detect link {link} is not accessible.')
    return link_status


'''checkmdfilesinfolder'''
def checkmdfilesinfolder(folder_path):
    result = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                print(f'Start checking {filepath}')
                result[filepath] = checklinksinfile(filepath)
    return result


'''DEBUG'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check whether all checkpoints in ModelZoo is available.')
    parser.add_argument('--dir', dest='dir', help='directory which contains .md files for demonstrating ModelZoo', required=True, type=str)
    parser.add_argument('--out', dest='out', help='output filepath for saving the results', required=False, type=str, default='modelzoo_available_check_results.pkl')
    args = parser.parse_args()
    results = checkmdfilesinfolder(args.dir)
    pickle.dump(results, open(args.out, 'wb'))