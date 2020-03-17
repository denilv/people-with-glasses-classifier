import os
import os.path as osp
import sys
import time
from argparse import ArgumentParser

import cv2
import torch

from modules.comp_tools import preprocessing_fn
from modules.utils import crop_img, find_face, open_img


def eprint(arg):
    print(arg, file=sys.stderr)


def default_argument_parser():
    parser = ArgumentParser(description="Cmd tool")
    parser.add_argument('model', help='Path to torchscript model')
    parser.add_argument('fld', help='Folder with images')
    parser.add_argument('--threshold', help='Decision threshold', type=float, default=0.5)
    return parser


class Predictor():

    def __init__(
        self, 
        model, 
        th=0.5, 
        shape=(120, 120),
        preprocessing_fn=preprocessing_fn,
    ):
        self.model = model
        self.th = th
        self.activation = torch.sigmoid
        self.shape = shape

    def predict(self, img):
        bbox = find_face(img)
        if bbox is None:
            # cannot find face on image
            return 0
        top, bot, left, right = bbox
        crop = img[top: bot, left: right] / 255.
        # crop = crop_img(img) / 255.
        crop = cv2.resize(crop, self.shape)
        crop = preprocessing_fn(crop)
        tensor = torch.Tensor(crop).unsqueeze(0)
        with torch.no_grad():
            logit = self.model(tensor)
        prob = self.activation(logit)
        return 1 if prob > self.th else 0


if __name__=='__main__':
    args = default_argument_parser().parse_args()
    model = torch.jit.load(args.model)
    predictor = Predictor(model, th=args.threshold)
    prediction_time = 0
    start = time.time()
    fnames = os.listdir(args.fld)
    files_seen = 0
    try:
        for fname in fnames:
            fp = osp.join(args.fld, fname)
            img = open_img(fp)
            prediction_start = time.time()
            prediction = predictor.predict(img)
            prediction_time += time.time() - prediction_start
            if prediction==1:
                print(fp)
            files_seen += 1
    except KeyboardInterrupt:
        eprint('Keyboard Interrupt') 
    elapsed_time = time.time() - start

    eprint(f'\nElapsed time:\t{elapsed_time:0.2f}\n')    
    
    eprint('FULL PIPELINE (open + crop face + predict):')
    full_fps = files_seen / elapsed_time
    full_time_per_img = 1.0 / full_fps
    eprint(f'FPS:\t\t{full_fps:0.2f}')
    eprint(f'Time per img:\t{full_time_per_img:0.3f}\n')

    eprint('PREDICT ONLY:')
    pred_fps = files_seen / prediction_time
    pred_time_per_img = 1.0 / pred_fps
    eprint(f'FPS:\t\t{pred_fps:0.2f}')
    eprint(f'Time per img:\t{pred_time_per_img:0.3f}')
