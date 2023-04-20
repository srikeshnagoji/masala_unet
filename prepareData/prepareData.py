import pandas as pd
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def get_dataset_dataframe(base_path: str, subset_count: int, train_val_split_ratio: float, train_test_split_ratio: float, seed: int, dataset_cat: str):
    data = []
    base_path = Path(base_path)
    for dir_ in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                data.append([dir_, img_path])
        else:
            print(f"[INFO] This is not a dir --> {dir_path}")


    df = pd.DataFrame(data, columns=["dir_name", "image_path"])

    df_imgs = df[~df["image_path"].str.contains("mask")].reset_index(drop=True)
    df_masks = df_imgs.copy() #df[df["image_path"].str.contains("mask")].reset_index(drop=True)

    # Mask Mapping correction..
    df_masks['image_path'] = df_imgs['image_path'].apply(lambda x: f'{x[:-len(".tif")]}_mask.tif')

    if subset_count != -1:
        df_imgs = df_imgs[:subset_count]
        df_masks = df_masks[:subset_count]

    print(f'Sample Image Path: {df_imgs["image_path"][0]}')
    print(f'Sample Mask Path: {df_masks["image_path"][0]}')

    # imgs = sorted(
    #     df_imgs["image_path"].values, key=lambda x: int((x[BASE_LEN:-END_LEN]))
    # )
    # masks = sorted(
    #     df_masks["image_path"].values,
    #     key=lambda x: int((x[BASE_LEN:-END_MASK_LEN])),
    # )

    imgs = list(df_imgs["image_path"].values)
    masks = list(df_imgs["image_path"].values)

    # print(imgs)
    # print(masks)

    dff = pd.DataFrame(
        # {"patient": df_imgs.dir_name.values, TODO: Removed this, add this extra info later when needed
            {"image_path": imgs, "mask_path": masks}
    )
    dff["diagnosis"] = dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

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
        df = train_df
    elif dataset_cat == 'Val':
        df = val_df
    elif dataset_cat == 'Test':
        df = test_df
    
    return df

def pos_neg_diagnosis(mask_path):
    """
    To assign 0 or 1 based on the presence of tumor.
    """
    val = np.max(np.asarray(Image.open(mask_path)))
    if val > 0:
        return 1
    else:
        return 0


def get_df_item(df, idx, transform_func):

    image = np.asarray(Image.open(df['image_path'].iloc[idx]).convert("RGB"))
    mask = np.asarray(Image.open(df['mask_path'].iloc[idx]).convert("L"))
    
    # image = np.array(cv2.imread(df['image_path'].iloc[idx]))
    # mask = np.array(cv2.imread(df['mask_path'].iloc[idx], 0))

    if transform_func:
        augmented = transform_func(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"]
    #         mask = np.expand_dims(augmented["mask"], axis=0)# Do not use this

    return image, mask
    pass