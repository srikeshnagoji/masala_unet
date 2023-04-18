import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lion_pytorch import Lion
import torch.optim as optim
from model.unet_attention import AttentionUNet
from model.unet_attention_with_ff import AttentionUNetFF
from model.unet_inception import InceptionUNet
from model.unet_attention_inception import InceptionAttentionUNet
from model.unet_attention_inception_ff import InceptionAttentionUNetFF
from model.unet3p_attention import UNet_3Plus_attn
from model.unet3p_attention_ff import UNet_3Plus_attn_FF
from model.UNet_3Plus_attn_FFAtDecoder import UNet_3Plus_attn_FFAtDecoder
from model.unet_inception_ff_at_decoder import InceptionUNetFFAtDecoder
from model.UNet_3Plus_attn_FF_enc_and_dec import UNet_3Plus_attn_FF_enc_and_dec
from model.unet3p import UNet3Plus
from model.unet.unet import UNet
from model.unet_attention_with_ff_se import AttentionUNetFFSE
from model.unet_attention_with_ff_3p import AttentionUNetFF3p
from model.unet_attention_with_ff_skip import AttentionUNetFFskip
from model.unet_inception_ff import InceptionUNetFF

from dataset import ISICDataset, GenericNpyDataset, BrainMRIDataset
from prepareData.prepareData import get_dataset_dataframe
from util.helper import pos_neg_diagnosis
from metrics.diceMetrics import dice_coef_metric, compute_iou, DiceLoss
from accelerate import Accelerator
import wandb

import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensor

from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


model_name = "unet_vanced"

train_loss = DiceLoss()
PATCH_SIZE = 128


# Parse CLI arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-slr", "--scale_lr", action="store_true", help="Whether to scale lr."
    )
    # TODO: implement other logging strategies in future
    parser.add_argument(
        "-rt",
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb"],
        help="Where to log to. Currently only supports wandb",
    )
    parser.add_argument(
        "-ld", "--logging_dir", type=str, default="logs", help="Logging dir."
    )
    parser.add_argument(
        "-od", "--output_dir", type=str, default="output", help="Output dir."
    )
    parser.add_argument(
        "-mp",
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to do mixed precision",
    )
    parser.add_argument(
        "-ga",
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="The number of gradient accumulation steps.",
    )
    parser.add_argument(
        "-img",
        "--img_folder",
        type=str,
        default="ISBI2016_ISIC_Part3B_Training_Data",
        help="The image file path from data_path",
    )
    parser.add_argument(
        "-csv",
        "--csv_file",
        type=str,
        default="ISBI2016_ISIC_Part3B_Training_GroundTruth.csv",
        help="The csv file to load in from data_path",
    )
    parser.add_argument(
        "-sc",
        "--self_condition",
        action="store_true",
        help="Whether to do self condition",
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "-ab1",
        "--adam_beta1",
        type=float,
        default=0.95,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "-ab2",
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "-aw",
        "--adam_weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay magnitude for the Adam optimizer.",
    )
    parser.add_argument(
        "-ae",
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "-ul", "--use_lion", type=bool, default=False, help="use Lion optimizer"
    )
    parser.add_argument(
        "-ic",
        "--mask_channels",
        type=int,
        default=1,
        help="input channels for training (default: 3)",
    )
    parser.add_argument(
        "-c",
        "--input_img_channels",
        type=int,
        default=3,
        help="output channels for training (default: 3)",
    )
    parser.add_argument(
        "-is",
        "--image_size",
        type=int,
        default=128,
        help="input image size (default: 128)",
    )
    parser.add_argument(
        "-dd", "--data_path", default="./data", help="directory of input image"
    )
    parser.add_argument("-d", "--dim", type=int, default=64, help="dim (default: 64)")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10000,
        help="number of epochs (default: 10000)",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=8,
        help="batch size to train on (default: 8)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="number of timesteps (default: 1000)",
    )
    parser.add_argument("-ds", "--dataset", default="generic", help="Dataset to use")
    parser.add_argument(
        "--save_every", type=int, default=100, help="save_every n epochs (default: 100)"
    )
    parser.add_argument(
        "--load_model_from", default=None, help="path to pt file to load from"
    )
    return parser.parse_args()


def load_data(args):
    # Load dataset
    if args.dataset == "ISIC":
        transform_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(transform_list)
        train_dataset = ISICDataset(
            args.data_path,
            args.csv_file,
            args.img_folder,
            transform=transform_train,
            training=True,
            flip_p=0.5,
        )
        val_dataset = None
        test_dataset = None
    elif args.dataset == "brain":
        # final dataframe
        BASE_PATH = args.data_path
        BASE_LEN = len(BASE_PATH) + len("/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_")
        END_LEN = len(".tif")  # image
        END_MASK_LEN = len("_mask.tif")  # mask

        IMG_SIZE = 512
        df = get_dataset_dataframe(BASE_PATH)

        df_imgs = df[~df["image_path"].str.contains("mask")]
        df_masks = df[df["image_path"].str.contains("mask")]
        imgs = sorted(
            df_imgs["image_path"].values, key=lambda x: int((x[BASE_LEN:-END_LEN]))
        )
        masks = sorted(
            df_masks["image_path"].values,
            key=lambda x: int((x[BASE_LEN:-END_MASK_LEN])),
        )
        dff = pd.DataFrame(
            {"patient": df_imgs.dir_name.values, "image_path": imgs, "mask_path": masks}
        )
        dff["diagnosis"] = dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

        print("Amount of patients: ", len(set(dff.patient)))
        print("Amount of records: ", len(dff))

        train_df, val_df = train_test_split(dff, stratify=dff.diagnosis, test_size=0.1)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_df, test_df = train_test_split(
            train_df, stratify=train_df.diagnosis, test_size=0.12
        )
        train_df = train_df.reset_index(drop=True)

        transform_train = A.Compose(
            [
                A.Resize(width=PATCH_SIZE, height=PATCH_SIZE, p=1.0),
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
        train_dataset = BrainMRIDataset(train_df, transform=transform_train)

        val_dataset = BrainMRIDataset(val_df, transform=transform_train)

        test_dataset = BrainMRIDataset(test_df, transform=transform_train)

    elif args.dataset == "generic":
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

    # Define PyTorch data generator
    train_data_generator = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_data_generator = None
    if val_dataset is not None:
        val_data_generator = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True
        )

    test_data_generator = None
    if test_dataset is not None:
        test_data_generator = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True
        )

    return train_data_generator, val_data_generator, test_data_generator


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


def main():
    args = parse_args()
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )
    if accelerator.is_main_process:
        # accelerator.init_trackers("experiment", config=vars(args))
        # accelerator.init_trackers("unet_vanced", config=vars(args))
        accelerator.init_trackers("unet_vanced_brain", config=vars(args))

    # DEFINE MODEL
    # model = UNet()
    # model = AttentionUNet(n_classes=1)
    # model = UNet3Plus()
    # model = UNet_3Plus_attn()
    # model = AttentionUNetFF(n_classes=1)
    # model = AttentionUNetFFSE(n_classes=1)
    # model = InceptionUNet(n_channels=3, n_classes=1)
    # model = InceptionAttentionUNet(n_channels=3, n_classes=1)
    # model = InceptionAttentionUNetFF(n_channels=3, n_classes=1)
    # model = AttentionUNetFF3p(n_classes=1)
    # model = InceptionUNetFF(n_channels=3, n_classes=1)
    # model = InceptionUNetFFAtDecoder(n_channels=3, n_classes=1)
    # model = UNet_3Plus_attn_FF()
    model = UNet_3Plus_attn_FFAtDecoder()
    # new
    # model = AttentionUNetFFskip(n_classes=1)

    # //NOT USED
    # model = Unet(
    #     dim=args.dim,
    #     image_size=args.image_size,
    #     dim_mults=(1, 2, 4, 8),
    #     mask_channels=args.mask_channels,
    #     input_img_channels=args.input_img_channels,
    #     self_condition=args.self_condition,
    # )

    # LOAD DATA
    train_data_generator, val_data_generator, test_data_generator = load_data(args)
    # training_generator = tqdm(data_loader, total=int(len(data_loader)))
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.batch_size
            * accelerator.num_processes
        )

    # Initialize optimizer
    if not args.use_lion:
        print("Using Adam Optimizer.")
        # optimizer = optim.SGD(
        #     model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001
        # )
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        print("Using LION Optimizer.")
        optimizer = Lion(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
        )

    # TRAIN MODEL
    counter = 0
    model, optimizer, train_data_loader = accelerator.prepare(
        model, optimizer, train_data_generator
    )
    print("using device: ", accelerator.device)
    model = model.to(accelerator.device)

    if args.load_model_from is not None:
        save_dict = torch.load(args.load_model_from)
        model.load_state_dict(save_dict["model_state_dict"])
        optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        accelerator.print(f"Loaded from {args.load_model_from}")

    # printing number of params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params of {model_name} model: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params of {model_name} model: {pytorch_total_params}")

    # take smaller steps as we reach minimum
    scheduler = ReduceLROnPlateau(optimizer, "min")  # TODO: uncomment this

    # Iterate across training loop
    least_loss = 9999999

    for epoch in tqdm(range(args.epochs)):
        running_loss = 0.0
        losses = []
        train_iou = []

        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        for img, mask in tqdm(train_data_loader):
            with accelerator.accumulate(model):
                img = img.to(accelerator.device)
                mask = mask.to(accelerator.device)

                outputs = model(img)

                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < 0.5)] = 0.0
                out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

                train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
                train_iou.append(train_dice)

                loss = train_loss(outputs, mask)
                losses.append(loss.item())

                # loss = diffusion(mask, img)
                accelerator.log({"Dice loss": loss.item()})  # Log loss to wandb
                accelerator.log({"Train IOU": train_dice})  # Log IOU to wandb
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * img.size(0)
        counter += 1
        epoch_loss = running_loss / len(train_data_generator)
        print("Epoch Loss : {:.4f}".format(epoch_loss))
        accelerator.log({"epoch_loss": epoch_loss})

        mean_dice_train = np.array(train_iou).mean()
        accelerator.log({"Mean IOU on train set": mean_dice_train})

        val_mean_iou = None
        if val_data_generator is not None:
            val_mean_iou = compute_iou(
                model, val_data_generator, device=accelerator.device
            )
            accelerator.log({"Mean IOU on validation set": val_mean_iou})

        mean_loss = np.array(losses).mean()
        accelerator.log({"Mean Dice loss (training epoch)": mean_loss})

        print(
            "Mean loss on train:",
            mean_loss,
            "\nMean DICE on train:",
            mean_dice_train,
            "\nMean DICE on validation:",
            val_mean_iou,
        )

        scheduler.step(mean_loss)  # TODO: uncomment this

        if mean_loss < least_loss:
            least_loss = mean_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # 'loss': loss.cpu().detach().numpy(),
                    "loss": loss,
                },
                os.path.join(checkpoint_dir + "/best", "best_model.pt"),
            )
        # INFERENCE

        if epoch % args.save_every == 0:
            # optimizer_to(optimizer, "cpu")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # 'loss': loss.cpu().detach().numpy(),
                    "loss": loss,
                },
                os.path.join(
                    checkpoint_dir, f"state_dict_epoch_{epoch}_loss_{epoch_loss}.pt"
                ),
            )
            # optimizer_to(optimizer, accelerator.device)

            pred = model(img)
            pred_out_cut = np.copy(pred.cpu().detach().numpy())
            pred_out_cut[np.nonzero(pred_out_cut < 0.5)] = 0.0
            pred_out_cut[np.nonzero(pred_out_cut >= 0.5)] = 1.0

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    # save just one image per batch
                    wandb.log(
                        {
                            "pred-img-mask": [
                                wandb.Image(img[0, 0, :, :]),
                                wandb.Image(mask[0, 0, :, :]),
                                wandb.Image(pred_out_cut[0, 0, :, :]),
                            ]
                        }
                    )


if __name__ == "__main__":
    main()
