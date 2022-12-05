import cv2
import torchvision.transforms as tfs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

def create_transforms(hydra_cfg, mode, degree=15):
    image_size = hydra_cfg.Dataset.image_size

    if mode == 'train':
        train_transforms = A.Compose([
                                    A.Affine(rotate=(-degree, degree), translate_percent=(0.05, 0.05), scale=(0.95, 1.05), cval=128),
                                    # A.HorizontalFlip(),
                                    # A.VerticalFlip(),
                                    # A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
                                    A.Resize(image_size, image_size),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                    ToTensorV2()
                                    ])
        return train_transforms

    else:   
        val_transforms = A.Compose([
                                    A.Resize(image_size, image_size),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                    ToTensorV2()
                                    ])
        return val_transforms

# def image_augmentation(image):
#     img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
#     image = img_aug(image)
#     return image
