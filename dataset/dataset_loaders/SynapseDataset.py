import os
import random
# import h5py
import numpy as np
import pandas as pd
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from prepareData.prepareData import get_dataset_dataframe

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensor

class SynapseDataset(Dataset):
    def __init__(self, train_val_split_ratio, train_test_split_ratio, seed, dataset_cat, base_path, subset_count, transform=None, training=True):
        self.base_path = str(base_path)
        self.subset_count = int(subset_count)
        self.transform = self.get_transform(transform)
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()

        BASE_LEN = len(base_path) + len("/images")
        END_LEN = len(".tif")  # image
        END_MASK_LEN = len("_mask.tif")  # mask

        df = get_dataset_dataframe(base_path)
        
        df_imgs = df[~df["image_path"].str.contains("mask")]
        df_masks = df[df["image_path"].str.contains("mask")]

        df_imgs = df_imgs[:subset_count]
        df_masks = df_imgs[:subset_count]
        
        imgs = sorted(df_imgs["image_path"].values)
        masks = sorted(df_masks["image_path"].values)
        # masks = sorted(
        #     df_masks["image_path"].values,
        #     key=lambda x: x#int((x[BASE_LEN:-END_MASK_LEN])),
        # )
        dff = pd.DataFrame(
            {"image_path": imgs, "mask_path": masks}
        )
        dff["diagnosis"] = dff["mask_path"].apply(lambda x: self.pos_neg_diagnosis(x))

        # print("Amount of patients: ", len(set(dff.patient)))
        print("Amount of records: ", len(dff))

        train_df, val_df = train_test_split(dff, stratify=dff.diagnosis, test_size=train_val_split_ratio, random_state=seed)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_df, test_df = train_test_split(
            train_df, stratify=train_df.diagnosis, test_size=train_test_split_ratio, random_state=seed
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        if dataset_cat == 'Train':
            self.df = train_df
        elif dataset_cat == 'Val':
            self.df = val_df
        elif dataset_cat == 'Test':
            self.df = test_df
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # if self.split == "train":
        #     slice_name = self.sample_list[idx].strip('\n')
        #     data_path = os.path.join(self.base_path, slice_name+'.npz')
        #     data = np.load(data_path)
        #     image, label = data['image'], data['label']
        # elif self.split == "test":
        #     slice_name = self.sample_list[idx].strip('\n')
        #     data_path = os.path.join(self.base_path, slice_name+'.npz')
        #     data = np.load(data_path, allow_pickle=True)
        #     image, label = data['image'], data['label']
        # # else:
        # #     vol_name = self.sample_list[idx].strip('\n')
        # #     filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
        # #     data = h5py.File(filepath)
        # #     image, label = data['image'][:], data['label'][:]

        # sample = {'image': image, 'label': label}
        # if self.transform:
        #     sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        # return sample

        image = cv2.imread(self.df['image_path'].iloc[idx])
        mask = cv2.imread(self.df['mask_path'].iloc[idx], 0)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)

            image = augmented["image"]
            mask = augmented["mask"]
        #         mask = np.expand_dims(augmented["mask"], axis=0)# Do not use this

        return image, mask
    
    def get_transform(self, transform_details):
        
        # Default Transform..
        if transform_details.DATASET_TRANSFORM_NAME == 'Default':
            transform_config = transform_details.DATASET_TRANSFORM_CONFIG
            transform_list = A.Compose(
                [
                    A.Resize(width=transform_config.image_size, height=transform_config.image_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25
                    ),
                    A.Normalize(p=1.0),
                    ToTensor(),
                ]
            )
        
        if transform_list:
            return transform_list
        
        return None

    def pos_neg_diagnosis(self,mask_path):
        """
        To assign 0 or 1 based on the presence of tumor.
        """
        val = np.max(cv2.imread(mask_path))
        if val > 0:
            return 1
        else:
            return 0