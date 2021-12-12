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
from face_model.face_gan import FaceGAN

class Segmentation2Face(object):
    def __init__(self, base_dir='./', size=1024, model=None, channel_multiplier=2, narrow=1, is_norm=True):
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier, narrow, is_norm)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, segf):
        # from segmentations to faces
        out = self.facegan.process(segf)

        return out
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPEN-Seg2face-512', help='GPEN model')
    parser.add_argument('--size', type=int, default=512, help='resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--is_norm', action='store_false', help='use sr or not')
    parser.add_argument('--indir', type=str, default='input/seg', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/outs-seg2face', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seg2face = Segmentation2Face(
        size=args.size,
        model=args.model,
        channel_multiplier=args.channel_multiplier,
        is_norm=False
    )

    files = sorted(glob.glob(os.path.join(args.indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)
        
        segf = cv2.imread(file, cv2.IMREAD_COLOR)

        realf = seg2face.process(segf)
        
        segf = cv2.resize(segf, realf.shape[:2])
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1])+'.jpg'), np.hstack((segf, realf)))
        
        if n%10==0: print(n, file)

if __name__=='__main__':
    main()
