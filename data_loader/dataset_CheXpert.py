import numpy as np
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
import os
from PIL import Image
import pandas as pd


def image_augmentation(image):
    img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
    image = img_aug(image)
    return image


class ChexpertDataset(Dataset):
    """Image generator
        Args:krurr
            dir_path (str): path to .csv file contains img paths and class labels
            mode (str, optional): define which mode you are using. Defaults to 'train'.
            use_frontal (bull) : 
    """
    def __init__(self, mode, root_path, folder_path, use_frontal, train_cols, use_enhancement, enhance_cols, enhance_time, flip_label, shuffle, seed, image_size, verbose):
        self.root_path = root_path
        self.folder_path = folder_path
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
        self.mode = mode

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


##############################
# dataset 선언 및 data loader
##############################

# trainset = ChexpertDataset(cfg.root_path, cfg.small_dir, cfg.mode, cfg.use_frontal, cfg.train_cols,cfg. use_enhancement, 
#                             cfg.enhance_cols, cfg.enhance_time, cfg.flip_label, cfg.shuffle, cfg.seed, cfg.image_size, cfg.verbose)
# testset = ChexpertDataset(cfg.root_path, cfg.small_dir, 'valid', cfg.use_frontal, cfg.train_cols,cfg. use_enhancement, 
#                             cfg.enhance_cols, cfg.enhance_time, cfg.flip_label, cfg.shuffle, cfg.seed, cfg.image_size, cfg.verbose)
# 
# trainloader =  torch.utils.data.DataLoader(trainset, batch_size=32, num_workers=2, drop_last=True, shuffle=True)
# testloader =  torch.utils.data.DataLoader(testset, batch_size=32, num_workers=2, drop_last=False, shuffle=False)


##############################
# data plot
##############################

# import matplotlib.pyplot as plt

# figure = plt.figure(figsize=(10, 10))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(trainset), size=(1,)).item()
#     img, label = trainset[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(f'Sample - {sample_idx} \n Labels - {label}')
#     plt.axis("off")
#     plt.imshow(img[0,:,:], cmap="gray")
# plt.show()


# for idx, data in enumerate(trainloader):
#     train_data, train_labels = data
#     print(len(train_data))
#     print(train_data.shape)
#     print('*********')
#     print(len(train_labels))
#     print(train_labels.shape)
#     break