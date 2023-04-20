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


def load_data(config):

    # Load Dataset (Generic Code):
    module = __import__(name=f'dataset.dataset_loaders.{config.DATASET.DATASET_LOADER}',fromlist = [f'{config.DATASET.DATASET_LOADER}'])

    train_dataloader = DataLoader(
        getattr(module, f'{config.DATASET.DATASET_LOADER}')(**vars(config.DATASET.DATASET_CONFIG), train_val_split_ratio=config.TRAIN.TRAIN_VAL_SPLIT, train_test_split_ratio=config.TEST.TRAIN_TEST_SPLIT, transform=config.DATASET.DATASET_TRANSFORMS, seed=config.SEED, dataset_cat='Train'),
        batch_size=config.TRAIN.BATCH_SIZE, 
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True)

    val_dataloader = None
    test_dataloader = None

    if config.TRAIN.TRAIN_VAL_SPLIT < 1:
        val_dataloader = DataLoader(
            getattr(module, f'{config.DATASET.DATASET_LOADER}')(**vars(config.DATASET.DATASET_CONFIG), train_val_split_ratio=config.TRAIN.TRAIN_VAL_SPLIT, train_test_split_ratio=config.TEST.TRAIN_TEST_SPLIT, transform=config.DATASET.DATASET_TRANSFORMS, seed=config.SEED, dataset_cat='Val'),
            batch_size=config.TRAIN.BATCH_SIZE, 
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True)
    
    if config.TEST.TRAIN_TEST_SPLIT < 1:
        # Split has to be made..
        test_dataloader = DataLoader(
            getattr(module, f'{config.DATASET.DATASET_LOADER}')(**vars(config.DATASET.DATASET_CONFIG), train_val_split_ratio=config.TRAIN.TRAIN_VAL_SPLIT, train_test_split_ratio=config.TEST.TRAIN_TEST_SPLIT, transform=config.DATASET.DATASET_TRANSFORMS, seed=config.SEED, dataset_cat='Test'),
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True)

    # train_dataset = ISICDataset(
    #     args.data_path,
    #     args.csv_file,
    #     args.img_folder,
    #     transform_config=transform_config,
    #     training=True,
    #     flip_p=0.5,
    # )

    # Load dataset
    # if args.dataset == "ISIC":
    #     # transform_list = [
    #     #     transforms.Resize((args.image_size, args.image_size)),
    #     #     transforms.ToTensor(),
    #     # ]
    #     # transform_train = transforms.Compose(transform_list)

    #     transform_train = A.Compose(
    #         [
    #             A.Resize(width=args.image_size, height=args.image_size, p=1.0),
    #             A.VerticalFlip(p=0.5),
    #             ToTensor(),
    #         ]
    #     )

    #     train_dataset = ISICDataset(
    #         args.data_path,
    #         args.csv_file,
    #         args.img_folder,
    #         transform_config=transform_config,
    #         training=True,
    #         flip_p=0.5,
    #     )
    #     val_dataset = None
    #     test_dataset = None

    """
    TODO: Add Generic Dataset As Well..
    
    if args.dataset == "generic":
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(transform_list)
        train_dataset = GenericNpyDataset(
            args.data_path, transform=transform_train, test_flag=False
        )
        val_dataset = None
        test_dataset = None
    else:
        raise NotImplementedError(
            f"Your dataset {args.dataset} hasn't been implemented yet."
        )
    

    """
    
    return train_dataloader, val_dataloader, test_dataloader


## MPS..

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)