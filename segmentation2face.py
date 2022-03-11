'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import argparse
import glob
import time
import numpy as np
from PIL import Image
import __init_paths
import __download_weights
from face_model.face_gan import FaceGAN

class Segmentation2Face(object):
    def __init__(self, base_dir='./', in_size=1024, out_size=None, model=None, channel_multiplier=2, narrow=1, key=None, is_norm=True, device='cuda'):
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, channel_multiplier, narrow, key, is_norm, device=device)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, segf, aligned=True):
        # from segmentations to faces
        out = self.facegan.process(segf)

        return out, [segf], [out]
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPEN-Seg2face-512', help='GPEN model')
    parser.add_argument('--in_size', type=int, default=1024, help='in resolution of GPEN')
    parser.add_argument('--out_size', type=int, default=None, help='out resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--is_norm', action='store_false', help='use sr or not')
    parser.add_argument('--indir', type=str, default='input/seg', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/outs-seg2face', help='output folder')
    parser.add_argument('--aligned', action='store_true', help='If input are aligned images')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seg2face = Segmentation2Face(
        in_size=args.in_size,
        out_size=args.out_size,
        model=args.model,
        channel_multiplier=args.channel_multiplier,
        is_norm=False
    )

    files = sorted(glob.glob(os.path.join(args.indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)
        
        segf = cv2.imread(file, cv2.IMREAD_COLOR)

        realf = seg2face.process(segf)
        
        segf, _, _= cv2.resize(segf, realf.shape[:2], args.aligned)
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1])+'.jpg'), np.hstack((segf, realf)))
        
        if n%10==0: print(n, file)

if __name__=='__main__':
    main()
