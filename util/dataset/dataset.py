import sys
import argparse
from config import get_config
import torch

from torch.utils.data import Dataset, DataLoader

sys.path.append("../")




def get_args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to run config file', )
    args = parser.parse_args()
    config = get_config(args)
    return config