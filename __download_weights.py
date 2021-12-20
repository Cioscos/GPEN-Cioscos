"""
This code needs to download weights if they are not inside weights folder
"""

import requests
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from tqdm.auto import tqdm

BLOCK_SIZE = 1024
WEIGHTS_PATH = Path('.\weights')
WEIGHTS_EXTENSIONS = ['.pth', '.ckpt']

weights_urls = {
    'RetinaFace-R50'         : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth',
    'ParseNet-latest'        : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth',
    'model_ir_se50'          : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/model_ir_se50.pth',
    'GPEN-BFR-512'           : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth',
    'GPEN-BFR-512-D'         : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512-D.pth',
    'GPEN-BFR-256'           : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-256.pth',
    'GPEN-BFR-256-D'         : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-256-D.pth',
    'GPEN-Colorization-1024' : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Colorization-1024.pth',
    'GPEN-Inpainting-1024'   : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Inpainting-1024.pth',
    'GPEN-Seg2face-512'      : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Seg2face-512.pth',
    'rrdb_realesrnet_psnr'   : 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/rrdb_realesrnet_psnr.pth'
}

def download_weight(url: str):
    weight_name = url.split('/')[-1]
    print(f'Downloading {weight_name}')
    site = urlopen(url)
    meta = site.info()
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(meta['Content-Length'])
    progress_bar = tqdm(total = total_size_in_bytes, unit='iB', unit_scale=True)
    with open(WEIGHTS_PATH / weight_name, 'wb') as f:
        for data in response.iter_content(BLOCK_SIZE):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry

def get_weights_paths(dir_path, files_extensions=WEIGHTS_EXTENSIONS, subdirs=False, return_Path_class=False):
    dir_path = Path (dir_path)
    result = []

    if dir_path.exists():
        if subdirs:
            gen = scantree(str(dir_path))
        else:
            gen = os.scandir(str(dir_path))

        for x in list(gen):
            if any([x.name.lower().endswith(ext) for ext in files_extensions]):
                result.append( x.path if not return_Path_class else Path(x.path) )
    return sorted(result)

def remove_suffix(text, suffix):
    return (text[:-len(suffix)], True) if text.endswith(suffix) and len(suffix) != 0 else (text, False)

if not WEIGHTS_PATH.exists():
    os.makedirs(WEIGHTS_PATH)

found_files_paths = [ filepath for filepath in get_weights_paths(WEIGHTS_PATH) ]
for i, path in enumerate(found_files_paths):
    for extension in WEIGHTS_EXTENSIONS:
        filename, stemmed = remove_suffix(path.split('\\')[-1], extension)
        if stemmed : break
    found_files_paths[i] = filename

download_urls = []

for weight in weights_urls.keys():
    if weight not in found_files_paths:
        download_urls.append(weights_urls[weight])

if len(download_urls) != 0:
    with ThreadPoolExecutor() as executor:
        executor.map(download_weight, download_urls)
else:
    print('All weights are already downloaded')

print('GPEN is compiling C++/CUDA Pytorch extensions. Please wait, it might take a while.')