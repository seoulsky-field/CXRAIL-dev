""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
import os
import pandas as pd

from .readers import create_reader

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50

def image_augmentation(image):
    img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
    image = img_aug(image)
    return image

class CheXpertDataset(Dataset):
    """Image generator
        Args:
            dir_path (str): path to .csv file contains img paths and class labels
            mode (str, optional): define which mode you are using. Defaults to 'train'.
            use_frontal (bull) : 
    """
    def __init__(self, root_path, folder_path, mode, use_frontal, train_cols, use_enhancement, enhance_cols, enhance_time, flip_label, shuffle, seed, image_size, verbose):
        self.root_path = root_path
        self.folder_path = folder_path
        self.mode = mode
        self.use_frontal = use_frontal
        self.train_cols = train_cols
        self.use_enhancement = use_enhancement
        self.enhance_cols = enhance_cols
        self.enhance_time = enhance_time 
        self.flip_label = flip_label
        self.shuffle = shuffle
        self.seed = seed
        self.image_size = image_size
        self.verbose = verbose

        # Load data from csv
        if self.mode == 'train':
            self.file = 'train.csv'
        else:
            self.file = 'valid.csv'
        self.dir_path = self.root_path + self.folder_path
        self.csv_path = os.path.join(self.dir_path, self.file)
        self.df = pd.read_csv(self.csv_path)

        # Use frontal
        if self.use_frontal == True:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal'] 
        
        # enhancement (upsampling)
        if self.use_enhancement:
            assert isinstance(self.enhance_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in self.enhance_cols:
                print ('Upsampling %s...'%col)
                for times in range(self.enhance_time):
                    sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # value mapping (policy)
        for col in self.df.columns.values:
            self.df[col].fillna(0, inplace=True)
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other','Fracture','Support Devices']:
                self.df[col].replace(-1, 0, inplace=True)  
            else:
                pass
                
        # dataset lenght
        self._num_images = len(self.df)

        # 0 --> -1
        if self.flip_label and len(self.train_cols) == 1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

        # shuffle data
        if self.shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]        

        # multi-label or one-label
        if len(self.train_cols) > 1:                                                                 # multi-label
            if verbose:
                print ('-'*30)
                print(f'{self.mode} Dataset')
                print ('Multi-label mode: True, Number of classes: [%d]'%len(self.train_cols))
                print ('-'*30)
            self.select_cols = self.train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(self.train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:                                                                                        # one-label
            self.select_cols = [self.train_cols[0]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        # image, target
        self._images_list =  [self.root_path + path for path in self.df['Path'].tolist()]
        if len(self.train_cols) == 1:
            self.targets = self.df[self.train_cols[0]].values[:].tolist()
        else:
            self.targets = self.df[self.train_cols].values.tolist()

        # check data imbalance
        if True:
            if len(self.train_cols) == 1:       # one-label        
                if self.flip_label == True:
                    negtive_value = -1
                else:
                    negtive_value = 0
                self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[negtive_value]+self.value_counts_dict[1])
                if verbose:
                    # print ('-'*30)
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[negtive_value]))
                    print ('%s(C): imbalance ratio is %.4f'%(self.select_cols[0], self.imratio ))
                    # print ('-'*30)
            else:                               # multi-label
                imratio_list = []
                for class_key, select_col in enumerate(self.train_cols):
                    try:
                        imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    except:                     # if all labels are consist of one value
                        if len(self.value_counts_dict[class_key]) == 1 :
                            only_key = list(self.value_counts_dict[class_key].keys())[0]
                            if only_key == 0:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 0     # no postive samples
                            else:    
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 1     # no negative samples
                            
                    imratio_list.append(imratio)
                    if verbose:
                        # print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                        print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                        # print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list


    @property        
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.train_cols)

    @property  
    def data_size(self):
        return self._num_images 

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        # if self.mode == 'train' :
        #     if self.transforms is None:
        #         image = self.image_augmentation(image)
        #     else:
        #         image = self.transforms(image)
        if self.mode == 'train':
            image = image_augmentation(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if len(self.train_cols) > 1: # multi-class mode
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        return image, label


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
