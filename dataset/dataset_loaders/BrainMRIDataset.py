import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data import Dataset
# import numpy as np
# import cv2
import albumentations as A
from albumentations.pytorch import ToTensor

from prepareData.prepareData import get_dataset_dataframe, get_df_item
# from util.helper import pos_neg_diagnosis
import pandas as pd

class BrainMRIDataset(Dataset):
    def __init__(self, train_val_split_ratio, train_test_split_ratio, seed, dataset_cat, base_path, subset_count, transform=None, training=True):
        self.base_path = str(base_path)
        self.subset_count = int(subset_count)
        self.transform = self.get_transform(transform)
        self.training = bool(training)

        # BASE_LEN = len(base_path) + len("/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_")
        # END_EXTENSION = len(".tif")  # image
        # END_MASK_LEN = len("_mask.tif")  # mask

        self.df = get_dataset_dataframe(base_path, subset_count, train_val_split_ratio, train_test_split_ratio,seed,dataset_cat)

        # df_imgs = df[~df["image_path"].str.contains("mask")].reset_index(drop=True)
        # df_masks = df_imgs.copy() #df[df["image_path"].str.contains("mask")].reset_index(drop=True)

        # # Mask Mapping correction..
        # df_masks['image_path'] = df_imgs['image_path'].apply(lambda x: f'{x[:-len(".tif")]}_mask.tif')

        # if subset_count != -1:
        #     df_imgs = df_imgs[:subset_count]
        #     df_masks = df_masks[:subset_count]

        # print(f'Sample Image Path: {df_imgs["image_path"][0]}')
        # print(f'Sample Mask Path: {df_masks["image_path"][0]}')

        # # imgs = sorted(
        # #     df_imgs["image_path"].values, key=lambda x: int((x[BASE_LEN:-END_LEN]))
        # # )
        # # masks = sorted(
        # #     df_masks["image_path"].values,
        # #     key=lambda x: int((x[BASE_LEN:-END_MASK_LEN])),
        # # )

        # imgs = list(df_imgs["image_path"].values)
        # masks = list(df_imgs["image_path"].values)

        # # print(imgs)
        # # print(masks)

        # dff = pd.DataFrame(
        #     # {"patient": df_imgs.dir_name.values,
        #      {"image_path": imgs, "mask_path": masks}
        # )
        # dff["diagnosis"] = dff["mask_path"].apply(lambda x: self.pos_neg_diagnosis(x))

        # # print("Amount of patients: ", len(set(dff.patient)))
        # print("Amount of records: ", len(dff))

        # train_df, val_df = train_test_split(dff, stratify=dff.diagnosis, test_size=train_val_split_ratio, random_state=seed)
        # train_df = train_df.reset_index(drop=True)
        # val_df = val_df.reset_index(drop=True)

        # train_df, test_df = train_test_split(
        #     train_df, stratify=train_df.diagnosis, test_size=train_test_split_ratio, random_state=seed
        # )
        # train_df = train_df.reset_index(drop=True)
        # test_df = test_df.reset_index(drop=True)

        # if dataset_cat == 'Train':
        #     self.df = train_df
        # elif dataset_cat == 'Val':
        #     self.df = val_df
        # elif dataset_cat == 'Test':
        #     self.df = test_df
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # image = cv2.imread(self.df['image_path'].iloc[idx])
        # mask = cv2.imread(self.df['mask_path'].iloc[idx], 0)

        # if self.transform:
        #     augmented = self.transform(image=image, mask=mask)

        #     image = augmented["image"]
        #     mask = augmented["mask"]
        #         mask = np.expand_dims(augmented["mask"], axis=0)# Do not use this
        image, mask = get_df_item(self.df,idx,self.transform)
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

    # def get_image_and_mask(self, idx):
    #     image = cv2.imread(self.df.loc[idx, "image_path"])
    #     mask = cv2.imread(self.df.loc[idx, "mask_path"], 0)
    #     PATCH_SIZE = 128
    #     trans_ = A.Compose(
    #         [
    #             A.Resize(width=PATCH_SIZE, height=PATCH_SIZE, p=1.0),
    #             ToTensor(),
    #         ]
    #     )

    #     augmented = trans_(image=image, mask=mask)

    #     image = augmented["image"]
    #     mask = augmented["mask"]

    #     return image, mask