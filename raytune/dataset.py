import os
import numpy as np
import pandas as pd
import sys
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from config import cfg

def get_dataloader(data_dir):
        # Get data loader
    df_tr_new = modify_csv(os.path.join(data_dir,'train.csv'))
    df_val_new = modify_csv(os.path.join(data_dir,'valid.csv'))
    train_dataset = ChexpertDataset(df_tr_new)
    val_dataset = ChexpertDataset(df_val_new)

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

# Define simple Dataset class
class ChexpertDataset(Dataset):
    def __init__(self, data_info: pd.DataFrame):

        self._label = data_info.View
        self._num_image = len(self._label)
        self._image_paths = data_info.Path
        
    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        basic_path = '/home/dataset/chexpert'
        image = cv2.imread(os.path.join(basic_path, self._image_paths[idx]), 0)
        image = Image.fromarray(image)
        image = np.array(image)
        image = transform(image)
        labels = self._label[idx]

        path = os.path.join(basic_path, self._image_paths[idx])
        
        return (image, labels)


# 이미지 전처리
def border_pad(image):
    h, w, c = image.shape
    image = np.pad(image, ((0, 224 - h), (0, 224 - w), (0, 0)),
                   mode='constant',
                   constant_values=128.0)
    return image


def fix_ratio(image):
    h, w, c = image.shape
    if h >= w:
        ratio = h * 1.0 / w
        h_ = 224
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = 224
        h_ = round(w_ / ratio)
    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image = border_pad(image)
    return image

def transform(image):
    assert image.ndim == 2, "image must be gray image"
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = fix_ratio(image)
    # augmentation for train or co_train

    # normalization
    image = image.astype(np.float32) - 128.0
    # vgg and resnet do not use pixel_std, densenet and inception use.
    image /= 64.0
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))
    return image


# csv전처리
def modify_csv(file_path):
    df = pd.read_csv(file_path)

    df['AP/PA'].fillna('Lateral', inplace=True)
    df = df[["Path", 'AP/PA']]
    
    df = df.drop(df[df['AP/PA']=='LL'].index)
    df = df.drop(df[df['AP/PA']=='RL'].index)

    title_mapping = {'AP': 0, 'PA': 1, 'Lateral': 2}
    df['AP/PA'] = df['AP/PA'].map(title_mapping)

    df.rename(columns = {'AP/PA' : 'View'}, inplace=True)
    return df

# if __name__ == "__main__":
