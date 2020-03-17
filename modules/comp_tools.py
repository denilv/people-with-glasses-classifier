import os
import os.path as osp

import albumentations as A
import cv2
import numpy as np
import pretrainedmodels
import torch
import torchvision as tv
from albumentations.augmentations.functional import normalize
from PIL import Image
from torch import nn


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def preprocessing_fn(x):
    return to_tensor(normalize(x, MEAN, STD, max_pixel_value=1.0))


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

AUGMENTATIONS_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Blur(),
        A.JpegCompression(),
    ], p=0.5),
    A.ToFloat(max_value=1)
], p=1)


def dummy(x): return x


def get_model(model_name, num_classes=2, pretrained='imagenet', load_weights=None):
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(pretrained=pretrained)

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    init_features = model.last_linear.in_features
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(init_features),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=init_features, out_features=init_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(init_features // 2, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=init_features // 2, out_features=num_classes),
    )

    if load_weights:
        print('Loading', load_weights)
        state_dict = torch.load(load_weights)
        print(model.load_state_dict(state_dict['model_state_dict']))

    return model


def get_tv_model(model_name, num_classes=2, pretrained='imagenet'):
    model = tv.models.__dict__[model_name](pretrained=True)
    init_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(init_features),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=init_features, out_features=init_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(init_features // 2, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=init_features // 2, out_features=num_classes),
    )
    return model


class ClsDataset:
    def __init__(
        self,
        df,
        img_prefix,
        binary=False,
        mode='multiclass',
        augmentations=None,
        img_size=None,
        n_channels=3,
        shuffle=True,
        preprocess_img=dummy,
    ):
        self.df = df
        self.img_prefix = img_prefix
        self.binary = binary
        self.mode = mode
        self.img_ids = df.crop_path
        self.labels = df.has_glasses.values
        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.preprocess_img = preprocess_img

    def __len__(self):
        return len(self.img_ids)

    def get_img(self, i):
        # img_path = osp.join(self.img_prefix, img_id)
        img_path = self.img_ids.iloc[i]
        if self.n_channels != 3:
            raise NotImplementedError('Not implemented')
        img = np.array(Image.open(img_path))
        if self.img_size:
            img = cv2.resize(img, self.img_size)
        return img

    def augm_img(self, img):
        pair = self.augmentations(image=img)
        return pair['image']

    def __getitem__(self, i):
        # img_id = self.img_ids[i]
        img = self.get_img(i)
        if self.augmentations:
            img = self.augm_img(img)
        img = img / 255.
        img = np.clip(img, 0, 1)
        row = self.df.iloc[i]
        has_glasses = row['has_glasses']
        if self.binary:
            targets = np.array([has_glasses], dtype=np.float32)
            targets_one_hot = np.zeros((2,))
            targets_one_hot[has_glasses] = 1
        else:
            if self.mode == 'multiclass':
                raise NotImplementedError('Do not know how to impement multiclass.')
                # targets_one_hot = np.array(defect_map + [int(not has_glasses)])
                # targets = np.argmax(targets_one_hot)
            elif self.mode == 'multilabel':
                raise NotImplementedError('Do not know how to impement multilabel.')
            else:
                raise Exception(f'Unknown mode - {self.mode}')
        return {
            'features': self.preprocess_img(img), 
            'targets': targets,
            'targets_one_hot': targets_one_hot,
        }
