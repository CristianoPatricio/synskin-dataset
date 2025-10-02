import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Resize(224, 224),
        A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ], additional_targets={
        **{f'mask{i}': 'mask' for i in range(6)}, 
        'border_mask': 'mask', 
        'lesion_mask': 'mask'
    })

def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ], additional_targets={
        **{f'mask{i}': 'mask' for i in range(6)}, 
        'border_mask': 'mask', 
        'lesion_mask': 'mask'
    })