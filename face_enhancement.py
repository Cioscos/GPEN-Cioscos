'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import re
import cv2
import argparse
import torch
from io import BytesIO
import numpy as np
import __init_paths
import __download_weights
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points
from DFLIMG.DFLJPG import DFLJPG

import warnings
warnings.filterwarnings('ignore')


class FaceEnhancement(object):
    def __init__(self, args, base_dir='./', in_size=512, out_size=None, model=None, use_sr=True, device='cuda'):
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, args.channel_multiplier, args.narrow, args.key, device=device)
        self.srmodel =  RealESRNet(base_dir, args.sr_model, args.sr_scale, args.tile_size, device=device)
        self.faceparser = FaceParse(base_dir, device=device)
        self.use_sr = use_sr
        self.in_size = in_size
        self.out_size = in_size if out_size is None else out_size
        self.threshold = 0.9
        self.alpha = args.alpha

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.in_size, self.in_size), inner_padding_factor, outer_padding, default_square)

    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def process(self, img, aligned=False):
        orig_faces, enhanced_faces = [], []
        if aligned:
            ef = self.facegan.process(img)
            orig_faces.append(img)
            enhanced_faces.append(ef)

            if self.use_sr:
                ef = self.srmodel.process(ef)

            return ef, orig_faces, enhanced_faces

        if self.use_sr:
            img_sr = self.srmodel.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])

        facebs, landms = self.facedetector.detect(img)
        
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.in_size, self.in_size))
            
            # enhance the face
            ef = self.facegan.process(of)
            
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            #tmp_mask = self.mask
            tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0]/255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw) < 100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            ef = cv2.addWeighted(ef, self.alpha, of, 1.-self.alpha, 0.0)
            
            if self.in_size!=self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

        full_mask = full_mask[:, :, np.newaxis]
        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr*(1-full_mask) + full_img*full_mask)
        else:
            img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)

        return img, orig_faces, enhanced_faces


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
    parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model')
    parser.add_argument('--key', type=str, default=None, help='key of GPEN model')
    parser.add_argument('--in_size', type=int, default=512, help='in resolution of GPEN')
    parser.add_argument('--out_size', type=int, default=None, help='out resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--narrow', type=float, default=1, help='channel narrow scale')
    parser.add_argument('--alpha', type=float, default=1, help='blending the results')
    parser.add_argument('--use_sr', action='store_true', help='use sr or not')
    parser.add_argument('--use_cuda', action='store_true', help='use cuda or not')
    parser.add_argument('--device_idx', type=int, default=None, help="Device's index to use")
    parser.add_argument('--sr_model', type=str, default='realesrnet', help='SR model')
    parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')
    parser.add_argument('--tile_size', type=int, default=0, help='tile size for SR to avoid OOM')
    parser.add_argument('--aligned', action='store_true', help='If input are aligned images')
    parser.add_argument('--indir', type=str, default='input/imgs', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/outs-enhanced', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Show available CUDA devices
    if args.use_cuda:
        if args.device_idx is None:
            print('\n\nSelect the device to use:')

            n_devices = torch.cuda.device_count()
            devices_idxs = []
            for i in range(n_devices):
                print(f"[{i}] {torch.cuda.get_device_name(i)}")
                devices_idxs.append(i)

            while True:
                gpu_index = int(input('Insert device index: '))
                if gpu_index not in devices_idxs:
                    print('Select a valid index from the list!')
                else:
                    break

        cuda_device = f"cuda:{gpu_index if args.device_idx is None else args.device_idx}"
        torch.cuda.set_device(gpu_index if args.device_idx is None else args.device_idx)

    else:
        print('Using CPU')

    faceenhancer = FaceEnhancement(
        args,
        in_size=args.in_size,
        out_size=args.out_size,
        model=args.model,
        use_sr=args.use_sr,
        device=cuda_device if args.use_cuda else 'cpu'
    )

    imgPaths = make_dataset(args.indir)

    for n, file in enumerate(imgPaths):
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

        im = cv2.imread(file, cv2.IMREAD_COLOR)  # BGR
        if not isinstance(im, np.ndarray): print(filename, 'error'); continue

        #Optional
        #im = cv2.resize(im, (0, 0), fx=1, fy=1)

        img, _, _ = faceenhancer.process(im, args.aligned)

        if is_dfl_image:
            _, buffer = cv2.imencode('.jpg', img)
            img_byte_arr = BytesIO(buffer)
        else:
            im = cv2.resize(im, img.shape[:2][::-1])
            cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + '.jpg'), img)

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

        if n % 10 == 0: print(n, filename)

if __name__ == '__main__':
    main()
