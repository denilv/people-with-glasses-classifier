TRAIN_IMAGES = 'data/crops'
TRAIN_CSV = 'data/all.csv'

TRAIN_FOLDS = (0, 1, 2,)
VALID_FOLDS = (3,)
EPOCHS = 25
LR = 1e-3
BATCH_SIZE = 256

CUDA_VISIBLE_DEVICES = '1'

ENCODER = 'mobilenetv2'
ACTIVATION = 'sigmoid'
BINARY = True
MODE = 'multiclass' # or 'multilabel'

CONTINUE = None

LOGDIR = f'logs/{ENCODER}'


import os

import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from catalyst.dl.callbacks import AccuracyCallback, F1ScoreCallback
from catalyst.dl.runner import SupervisedRunner
from torch.utils.data import DataLoader as BaseDataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm.auto import tqdm

from modules.common import rle_decode
from modules.comp_tools import (AUGMENTATIONS_TRAIN, ClsDataset, get_model,
                                get_tv_model, preprocessing_fn, to_tensor)
from modules.mobilenetv2 import MobileNetV2

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


df = pd.read_csv(TRAIN_CSV).fillna('')
train_df = df[df.fold_num.isin(TRAIN_FOLDS)]
valid_df = df[df.fold_num.isin(VALID_FOLDS)]

print(df.columns)

model = MobileNetV2(num_classes=1)
if CONTINUE:
    print('Loading', CONTINUE)
    model.load_state_dict(torch.load(CONTINUE, map_location='cpu')['model_state_dict'])

train_dataset = ClsDataset(
    train_df,
    img_size=(120, 120),
    img_prefix=TRAIN_IMAGES, 
    augmentations=AUGMENTATIONS_TRAIN, 
    preprocess_img=preprocessing_fn,
    mode=MODE,
    binary=BINARY,
)
valid_dataset = ClsDataset(
    valid_df,
    img_size=(120, 120),
    img_prefix=TRAIN_IMAGES, 
    augmentations=None,
    preprocess_img=preprocessing_fn,
    mode=MODE,
    binary=BINARY,
)
train_dl = BaseDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dl = BaseDataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# experiment setup
num_epochs = EPOCHS
logdir = LOGDIR
loaders = {
    "train": train_dl,
    "valid": valid_dl
}
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 19, 22])

callbacks = [
    AccuracyCallback(num_classes=2, activation='Sigmoid', threshold=0.5),
]
runner = SupervisedRunner()

# Train
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=1,
    scheduler=scheduler,
    # main_metric='f1_score',
    minimize_metric=False,
)