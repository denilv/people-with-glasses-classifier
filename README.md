# people-with-glasses-classifier
Simple classifier of people wearing glasses

## Dataset preparation

Datasets:
* CelebFaces Attributes (CelebA) Dataset (https://www.kaggle.com/jessicali9530/celeba-dataset)
* MeGlass Dataset (https://github.com/cleardusk/MeGlass)
* SoF Dataset (https://sites.google.com/view/sof-dataset)
* SPECFACE Dataset (https://sites.google.com/site/specfaceiitkgp)

Statistics
| dataset  | has_glasses |  count |
|----------|-------------|:------:|
| celeba   | 0           | 189406 |
| celeba   | 1           | 13193  |
| meglass  | 0           | 33085  |
| meglass  | 1           | 14832  |
| sof      | 1           | 2428   |
| specface | 1           | 320    |

## Training

Binary classification with sigmod actication function.

__Model__: `MobileNetV2`  
__Optimizer__: `Adam(lr=1e-3)`  
__Scheduler__: `MultiStepLR(optimizer, milestones=[15, 19, 22])`  
__Input resolution__: `120x120`  
__Epochs__: `25`


__Augmentations__:
* HorizontalFlip
* VerticalFlip
* RandomBrightnessContrast
* ShiftScaleRotate
* Blur
* JpegCompression  
More details at: modules/comp_tools.py:37

### Model selection:
In general, I considered finetuning of 3 different models:
* Resnet18 (imagenet pretrained)
* MobileNetV2 (imagenet pretrained)
* MobileNetV3 (imagenet pretrained)

ResNet18 checkpoint is way bigger than 3 Mb (~45 Mb)  
MobileNetV2 checkpoint is about 9 Mb  
MobileNetV2 checkpoint is about 19 Mb  

### Model parameters compression

Post training static quantization.  
CPU inference speed up: ~8.7 times (83.9 fps -> 728.53 fps)  
Volume reductiuon: ~3.4 times (9.1 Mb -> 2.7 Mb)  

Source: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html 

## Validation

__StratifiedKFold__ for __5__ splits  
Folds #0, 1, 2 folds are using for _training_  
Fold #3 is using for _validation_  
Fold #4 is using for _test_  

### Per fold stats
```
fold_num  dataset   has_glasses    count
----------------------------------------
0         celeba    0              37899
                    1               2621
          meglass   0               7764
                    1               1820
          sof       1                486
          specface  1                 64
1         celeba    0              37861
                    1               2659
          meglass   0               7664
                    1               1920
          sof       1                486
          specface  1                 64
2         celeba    0              37953
                    1               2567
          meglass   0               7791
                    1               1792
          sof       1                486
          specface  1                 64
3         celeba    0              37892
                    1               2628
          meglass   0               5975
                    1               3608
          sof       1                485
          specface  1                 64
4         celeba    0              37801
                    1               2718
          meglass   0               3891
                    1               5692
          sof       1                485
          specface  1                 64
```

### Validation
```
              precision    recall  f1-score   support

           0       1.00      0.99      0.99     43867
           1       0.95      0.99      0.97      6785

    accuracy                           0.99     50652
   macro avg       0.97      0.99      0.98     50652
weighted avg       0.99      0.99      0.99     50652

Elapsed time: 19.74
0.00039 sec/img
2565.45 img/sec (fps)
```

### Test
```
              precision    recall  f1-score   support

           0       1.00      0.99      0.99     41692
           1       0.97      0.99      0.98      8958

    accuracy                           0.99     50650
   macro avg       0.98      0.99      0.99     50650
weighted avg       0.99      0.99      0.99     50650

Elapsed time: 18.59
0.00037 sec/img
2725.24 img/sec (fps)
```

More validation metrics: notebooks/validation.ipynb

## Inference

Hardware:
* i7-4700
* 48 GB RAM
* 2xGTX1080ti  

Steps:
1. Read image
2. Try to find faces with __dlib__
3. If no faces were found then return negative prediction
4. Crop face
5. Resize face crop
6. Model inference

Time on cpu:
* Full pipeline loop: from 0.01 to 0.1 sec/img (depends on original image resolution)
* Prediction only: 0.003 sec/img

## Run

1. `cd /.../people-with-glasses-classifier`
2. `python -m venv env`
3. `source env/bin/activate`
4. `pip install -r requirements.txt`
5. `PYTHONPATH=./ python scripts/main.py trained_models/quantized-mobilenetv2-scripted.pth examples/without_glasses`

Usage:
```bash
usage: main.py [-h] [--threshold THRESHOLD] [--use-gpu] model fld

Cmd tool

positional arguments:
  model                 Path to torchscript model
  fld                   Folder with images

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Decision threshold

```