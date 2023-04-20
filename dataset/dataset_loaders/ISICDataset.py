import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensor

from prepareData.prepareData import get_dataset_dataframe, get_df_item

class ISICDataset(Dataset):
    def __init__(
        self, train_val_split_ratio, train_test_split_ratio, seed, dataset_cat, base_path, subset_count, transform=None, training=True
    ):
        # self.df = pd.read_csv(os.path.join(base_path, csv_file_name), encoding="gbk")
        # self.img_folder_name = img_folder_name
        # self.name_list = df.iloc[:, 0].tolist()
        # self.label_list = df.iloc[:, 1].tolist()
        self.base_path = base_path
        self.transform = self.get_transform(transform)
        self.training = bool(training)

        self.df = get_dataset_dataframe(base_path, subset_count, train_val_split_ratio, train_test_split_ratio,seed,dataset_cat)

        pass

    def __len__(self):
        # return len(self.name_list)
        return len(self.df)

    def __getitem__(self, idx):
        # """Get the images"""
        # name = self.name_list[index] + ".jpg"
        # img_path = os.path.join(self.base_path, self.img_folder_name, name)

        # mask_name = name.split(".")[0] + "_Segmentation.png"
        # msk_path = os.path.join(self.base_path, self.img_folder_name, mask_name)

        # img = Image.open(img_path).convert("RGB")
        # mask = Image.open(msk_path).convert("L")

        # if self.training:
        #     label = 0 if self.label_list[index] == "benign" else 1
        # else:
        #     label = int(self.label_list[index])

        # # if self.transform:
        # #     # save random state so that if more elaborate transforms are used
        # #     # the same transform will be applied to both the mask and the img
        # #     state = torch.get_rng_state()
        # #     img = self.transform(img)
        # #     torch.set_rng_state(state)
        # #     mask = self.transform(mask)
        # #     if random.random() < self.flip_p:
        # #         img = F.vflip(img)
        # #         mask = F.vflip(mask)

        # if self.transform:
        #     augmented = self.transform(image=image, mask=mask)

        #     image = augmented["image"]
        #     mask = augmented["mask"]

        # return (image, mask, label)

        image, mask = get_df_item(self.df,idx,self.transform)
        # print(type(image), type(mask))
        # print(image.min(), image.max())
        # print(mask.min(), mask.max())
        return image, mask
    
    def get_transform(self, transform_details):
        
        # Default Transform..
        if transform_details.DATASET_TRANSFORM_NAME == 'Default':
            transform_config = transform_details.DATASET_TRANSFORM_CONFIG
            transform_list = A.Compose(
                [
                    A.Resize(width=transform_config.image_size, height=transform_config.image_size, p=1.0),
                    A.VerticalFlip(p=0.5),
                    ToTensor(),
                ]
            )
        
        if transform_list:
            return transform_list
        
        return None