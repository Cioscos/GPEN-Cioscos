'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import re
import cv2
import argparse
from io import BytesIO
import __init_paths
import __download_weights
from face_model.face_gan import FaceGAN
from DFLIMG.DFLJPG import DFLJPG

class FaceColorization(object):
    def __init__(self, base_dir='./', in_size=1024, out_size=1024, model=None, channel_multiplier=2, narrow=1, key=None, device='cuda'):
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, channel_multiplier, narrow, key, device=device)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, gray, aligned=True):
        # colorize the face
        out = self.facegan.process(gray)

        return out, [gray], [out]
        
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def alphaNumOrder(string):
    return ''.join([format(int(x), '05d') if x.isdigit()
                    else x for x in re.split(r'(\d+)', string)])

def make_dataset(dirs):
    images = []
    assert os.path.isdir(dirs), '%s is not a valid directory' % dirs

    for root, _, fnames in os.walk(dirs):
        for fname in sorted(fnames, key=alphaNumOrder):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPEN-Colorization-1024', help='GPEN model')
    parser.add_argument('--in_size', type=int, default=1024, help='in resolution of GPEN')
    parser.add_argument('--out_size', type=int, default=None, help='out resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--indir', type=str, default='input/grays', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/outs-colorization', help='output folder')
    parser.add_argument('--aligned', action='store_true', help='If input are aligned images')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    facecolorizer = FaceColorization(
        in_size=args.in_size,
        out_size=args.out_size,
        model=args.model,
        channel_multiplier=args.channel_multiplier
    )

    imgPaths = make_dataset(args.indir)

    for n, file in enumerate(imgPaths):
        # Take DFL metadata
        InputDflImg = DFLJPG.load(file)
        if not InputDflImg or not InputDflImg.has_data():
            print('\t################ No landmarks in file {}'.format(file))
            is_dfl_image = False
        else:
            is_dfl_image = True
            Landmarks = InputDflImg.get_landmarks()
            InputData = InputDflImg.get_dict()
            if InputDflImg.has_seg_ie_polys():
                xseg_polys = InputDflImg.get_seg_ie_polys()

        filename = os.path.basename(file)
        
        grayf = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        grayf = cv2.cvtColor(grayf, cv2.COLOR_GRAY2BGR) # channel: 1->3

        colorf, _, _= facecolorizer.process(grayf, args.aligned)
        colorf = cv2.resize(colorf, (grayf.shape[1], grayf.shape[0]))
 
        if is_dfl_image:
            _, buffer = cv2.imencode('.jpg', colorf)
            img_byte_arr = BytesIO(buffer)
        else:
            cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1])+'.jpg'), colorf)
        
        if is_dfl_image:
            OutputDflImg = DFLJPG.load(
                os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + '.jpg'),
                image_as_bytes=img_byte_arr.getvalue()
            )
            OutputDflImg.set_dict(InputData)
            OutputDflImg.set_landmarks(Landmarks)
            if InputDflImg.has_seg_ie_polys():
                OutputDflImg.set_seg_ie_polys(xseg_polys)
            OutputDflImg.save()
        
        if n%10==0: print(n, file)

if __name__=='__main__':
    main()
