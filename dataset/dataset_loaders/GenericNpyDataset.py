import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch

# TODO: Condition to Standard..

class GenericNpyDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, transform, test_flag: bool = True):
        """
        Genereic dataset for loading npy files.
        The npy store 3D arrays with the first two dimensions being the image and the third dimension being the channels.
        channel 0 is the image and the other channel is the label.
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag
        self.filenames = [x for x in os.listdir(self.directory) if x.endswith(".npy")]

    def __getitem__(self, x: int):
        fname = self.filenames[x]
        npy_img = np.load(os.path.join(self.directory, fname))
        img = npy_img[:, :, :1]
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = npy_img[:, :, 1:]
        mask = np.where(mask > 0, 1, 0)
        image = img[:, ...]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        if self.transform:
            # save random state so that if more elaborate transforms are used
            # the same transform will be applied to both the mask and the img
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)
        if self.test_flag:
            return image, mask, fname
        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)
