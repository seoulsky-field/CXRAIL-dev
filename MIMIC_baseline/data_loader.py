import numpy as np
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
import os
from PIL import Image
import pandas as pd



class CXRDataset(Dataset):
    """Image generator
        Args:
            dir_path (str): path to .csv file contains img paths and class labels
            mode (str, optional): define which mode you are using. Defaults to 'train'.
            use_frontal (bull) : 
    """
    def __init__(self, 
                 #input
                 mode, 
                 dataset,
                 labeler,
                 transforms,

                 #hydra
                 root_path, folder_path, image_size, labeler_path, #default settings
                 shuffle, seed, verbose, #experiment settings
                 use_frontal, use_enhancement, enhance_time, flip_label,
                 train_cols, enhance_cols
                 ):
        self.dataset = dataset
        self.mode = mode
        self.labeler = labeler
        # path
        self.root_path = root_path
        self.folder_path = folder_path
        self.labeler_path = labeler_path

        #columms
        self.train_cols =train_cols
        self.enhance_cols = enhance_cols

        # setting
        self.seed = seed
        self.image_size = image_size
        self.use_frontal = use_frontal
        self.use_enhancement = use_enhancement
        self.enhance_time = enhance_time 
        self.flip_label = flip_label
        self.shuffle = shuffle
        self.verbose = verbose
        self.transforms = transforms
        self.dir_path = self.root_path + self.folder_path
        

        # Choose Dataset
        if self.dataset == 'CheXpert':
            self.df = pd.read_csv(os.path.join(self.root_path + self.folder_path, self.mode+'.csv'))
        elif self.dataset == 'MIMIC':
            self.df = pd.read_csv(os.path.join(self.labeler_path, self.labeler, self.mode+'.csv')) # label이 지금은 임시 directory에 있기 labeler path를 따로 인자로 받고있는데, 나중에 수정가능 할 것 같습니다.

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
            np.random.seed(self.seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]        

        # multi-label or one-label
        if len(self.train_cols) > 1:                                                                 # multi-label
            if self.verbose == 1:
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
                if self.verbose == 1:
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
                    if self.verbose == 1:
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
        # transform -> albumentations
        image = cv2.imread(self._images_list[idx], 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
                 
        if len(self.train_cols) > 1: # multi-class mode
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        return image, label
