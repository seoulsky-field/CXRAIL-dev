import torch
import torchvision.transforms as tfs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


def create_transforms(Dataset_cfg, mode, ra_params=None):

    image_size = Dataset_cfg.image_size
    augmentation_mode = Dataset_cfg.augmentation_mode

    if mode == "train":
        if augmentation_mode == "auto":
            train_transforms = tfs.Compose(
                [
                    tfs.ToTensor(),
                    tfs.ConvertImageDtype(torch.uint8),
                    tfs.AutoAugment(tfs.AutoAugmentPolicy.IMAGENET),
                    tfs.ConvertImageDtype(torch.float32),
                    tfs.Resize((image_size, image_size)),
                    tfs.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        elif augmentation_mode == "random":
            # Reference: https://pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html
            train_transforms = tfs.Compose(
                [
                    tfs.ToTensor(),
                    tfs.ConvertImageDtype(torch.uint8),
                    tfs.RandAugment(**ra_params),
                    tfs.ConvertImageDtype(torch.float32),
                    tfs.Resize((image_size, image_size)),
                    tfs.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        elif augmentation_mode == "custom":
            train_transforms = A.Compose(
                [
                    # Below is sample implementation of customized usage
                    # A.Affine(
                    #     rotate=(-degree, degree),
                    #     translate_percent=(0.05, 0.05),
                    #     scale=(0.95, 1.05),
                    #     cval=128,
                    # ),
                    # A.HorizontalFlip(),
                    # A.VerticalFlip(),
                    # A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
                    A.Resize(image_size, image_size),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0,
                        always_apply=False,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            raise ValueError(f"Augmentation mode [{augmentation_mode}] is invalid")

        return train_transforms

    else:
        val_transforms = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    always_apply=False,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )

        return val_transforms
