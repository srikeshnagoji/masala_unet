import numpy as np
import cv2
import sys

sys.path.append("../")


from torch.optim.lr_scheduler import ReduceLROnPlateau


def pos_neg_diagnosis(mask_path):
    """
    To assign 0 or 1 based on the presence of tumor.
    """
    val = np.max(cv2.imread(mask_path))
    if val > 0:
        return 1
    else:
        return 0
