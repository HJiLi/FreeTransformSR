# FreeTransformSR

## Dependencies
- Python 3.9
- PyTorch 1.10.0

```
cd code
pip install -r requirements.txt
python setup.py develop
```
## Datasets
- FreeTransformSR

|  Training Set   | Testing Set   |
|  ----  | ----  |
|  DIV2K | Set5 + Set14 + BSD100 + Urban100 + Manga109  |

## Implementation of FreeTransformSR
### Train

```shell
#scale factor 2
python basicsr/train.py -opt options/train/FreeTransformSR/FreeTransformSR_x2.yml
#scale factor 3
python basicsr/train.py -opt options/train/FreeTransformSR/FreeTransformSR_x3.yml
#scale factor 4
python basicsr/train.py -opt options/train/FreeTransformSR/FreeTransformSR_x4.yml
```
### Test
```shell
#scale factor 2
python scripts/test_SISR.py -opt options/test/FreeTransformSR/FreeTransformSR_x2.yml --model_path ./experiments/pretrained_models/FreeTransformSR_x2.pth --save_img
#scale factor 3
python scripts/test_SISR.py -opt options/test/FreeTransformSR/FreeTransformSR_x3.yml --model_path ./experiments/pretrained_models/FreeTransformSR_x3.pth --save_img    
#scale factor 4
python scripts/test_SISR.py -opt options/test/FreeTransformSR/FreeTransformSR_x4.yml --model_path ./experiments/pretrained_models/FreeTransformSR_x4.pth --save_img  
```

### Download links
FreeTransformSR:
| Dataset  | Description |
| -------- | -------- |
| [DIV2K Train HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)   | Train Data (HR images)   |
| [DIV2K Train LR x2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip)   | Train Data x2 (LR images)   |
| [DIV2K Train LR x3](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip)   | Train Data x3 (LR images)    |
| [DIV2K Train LR x4](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip)   | Train Data x4 (LR images)   |
| [Test](https://1drv.ms/f/c/e93b322ec9a3379f/IgAumJO9B3BdTq1tuJFd8j2KARdYkGh4LY7gSWLDdlx7jNU)   | Testsets for Set5+Set14+BSD100+Urban100+Manga109|

