#####Data#####
import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

#configuration 할당
#from cfg import cfg (outdated as we use hydra)
import hydra

class ImageDataset(Dataset):
    def __init__(self, data_info: pd.DataFrame, cfg):
        """Image generator
        Args:
            dfat_info (pd.DataFrame): DataFrame that contains image paths and class labels
            cfg (str): configuration file.
            mode (str, optional): define which mode you are using. Defaults to 'train'.
        """
        
        self._label = data_info.label
        self._num_image = len(self._label)
        self._image_paths = data_info.Path
        self._cfg = cfg
        
        
    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        basic_path = self._cfg.root
        image = cv2.imread(os.path.join(basic_path, self._image_paths[idx]), 0)
        image = Image.fromarray(image)
        # if self._mode == 'train':
        #     image = GetTransforms(image, type=self.use_transforms_type)
        image = np.array(image)
        image = transform(image)
        labels = self._label[idx]
        path = self._image_paths[idx]

        return (image, labels)
    



def transform(image):
    assert image.ndim == 2, "image must be gray image"
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # normalization
    image = image.astype(np.float32) - 128.0
    image /= 64.0
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))

    return image


def assign_label(train_csv, valid_csv):
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    train_df['label'] = train_df.apply(lambda x: 0 if x['AP/PA']== 'AP' else (1 if x['AP/PA']== 'PA' else 2), axis=1)
    valid_df['label'] = valid_df.apply(lambda x: 0 if x['AP/PA']== 'AP' else (1 if x['AP/PA']== 'PA' else 2), axis=1)
    
    return train_df, valid_df



def chexpert_loader(train_df, valid_df, cfg):
    train_dataset = ImageDataset(train_df, cfg)
    val_dataset = ImageDataset(valid_df, cfg)

    train_loader = DataLoader(train_dataset,
                          batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers,
                          shuffle=True,
                          drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            shuffle=False,
                            drop_last=False)

    return train_loader, val_loader

