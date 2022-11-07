import cv2
import torchvision.transforms as tfs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def create_train_transforms(hydra_cfg):
    train_transforms = A.Compose([
                                A.Affine(rotate=(-15, 15), translate_percent=(0.05, 0.05), scale=(0.95, 1.05), cval=128),
                                # A.HorizontalFlip(),
                                # A.VerticalFlip(),
                                # A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
                                A.Resize(hydra_cfg.Dataset.image_size, hydra_cfg.Dataset.image_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])
    return train_transforms

def create_val_transforms(hydra_cfg):
    val_transforms = A.Compose([
                                A.Resize(hydra_cfg.Dataset.image_size, hydra_cfg.Dataset.image_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])
    return val_transforms

# def image_augmentation(image):
#     img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
#     image = img_aug(image)
#     return image
