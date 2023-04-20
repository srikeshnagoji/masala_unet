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
    def __init__(
        self,
        train_val_split_ratio,
        train_test_split_ratio,
        seed,
        dataset_cat,
        base_path,
        subset_count,
        transform=None,
        training=True,
    ):
        self.base_path = str(base_path)
        self.subset_count = int(subset_count)
        self.transform = self.get_transform(transform)
        self.training = bool(training)

        self.df = get_dataset_dataframe(
            base_path,
            subset_count,
            train_val_split_ratio,
            train_test_split_ratio,
            seed,
            dataset_cat,
        )

        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image, mask = get_df_item(self.df, idx, self.transform)
        return image, mask

    def get_transform(self, transform_details):
        # Default Transform..
        if transform_details.DATASET_TRANSFORM_NAME == "Default":
            transform_config = transform_details.DATASET_TRANSFORM_CONFIG
            transform_list = A.Compose(
                [
                    A.Resize(
                        width=transform_config.image_size,
                        height=transform_config.image_size,
                        p=1.0,
                    ),
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
