import os
import random
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
# import torchvision.transforms as transforms
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lion_pytorch import Lion

import pickle

import time

# from model.unet_attention import AttentionUNet
# from model.unet_attention_with_ff import AttentionUNetFF
# from model.unet_inception import InceptionUNet
# from model.unet_attention_inception import InceptionAttentionUNet
# from model.unet_attention_inception_ff import InceptionAttentionUNetFF
# from model.unet3p_attention import UNet_3Plus_attn

# from model.UNet_3Plus_attn_FF import UNet_3Plus_attn_FF

# from model.unet3p import UNet3Plus
# from model.unet.unet import UNet
# from model.unet_attention_with_ff_se import AttentionUNetFFSE
# from model.unet_attention_with_ff_3p import AttentionUNetFF3p
# from model.unet_attention_with_ff_skip import AttentionUNetFFskip
# from model.unet_inception_ff import InceptionUNetFF

# from dataset.dataset_loaders import ISICDataset, GenericNpyDataset, BrainMRIDataset
from metrics.diceMetrics import dice_coef_metric, compute_iou, DiceLoss
from accelerate import Accelerator
import wandb

from torch import backends

import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensor

import torch.nn.functional as F
from util.helper import get_args_config, optimizer_to, load_data

# model_name = "unet_vanced"

train_loss = DiceLoss()

# Parse CLI arguments
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-slr", "--scale_lr", action="store_true", help="Whether to scale lr."
#     )
#     parser.add_argument(
#         "-rt",
#         "--report_to",
#         type=str,
#         default="wandb",
#         choices=["wandb"],
#         help="Where to log to. Currently only supports wandb",
#     )
#     parser.add_argument(
#         "-ld", "--logging_dir", type=str, default="logs", help="Logging dir."
#     )
#     parser.add_argument(
#         "-od", "--output_dir", type=str, default="output", help="Output dir."
#     )
#     parser.add_argument(
#         "-mp",
#         "--mixed_precision",
#         type=str,
#         default="no",
#         choices=["no", "fp16", "bf16"],
#         help="Whether to do mixed precision",
#     )
#     parser.add_argument(
#         "-ga",
#         "--gradient_accumulation_steps",
#         type=int,
#         default=4,
#         help="The number of gradient accumulation steps.",
#     )
#     parser.add_argument(
#         "-img",
#         "--img_folder",
#         type=str,
#         default="ISBI2016_ISIC_Part3B_Training_Data",
#         help="The image file path from data_path",
#     )
#     parser.add_argument(
#         "-csv",
#         "--csv_file",
#         type=str,
#         default="ISBI2016_ISIC_Part3B_Training_GroundTruth.csv",
#         help="The csv file to load in from data_path",
#     )
#     parser.add_argument(
#         "-sc",
#         "--self_condition",
#         action="store_true",
#         help="Whether to do self condition",
#     )
#     parser.add_argument(
#         "-lr", "--learning_rate", type=float, default=5e-4, help="learning rate"
#     )
#     parser.add_argument(
#         "-ab1",
#         "--adam_beta1",
#         type=float,
#         default=0.95,
#         help="The beta1 parameter for the Adam optimizer.",
#     )
#     parser.add_argument(
#         "-ab2",
#         "--adam_beta2",
#         type=float,
#         default=0.999,
#         help="The beta2 parameter for the Adam optimizer.",
#     )
#     parser.add_argument(
#         "-aw",
#         "--adam_weight_decay",
#         type=float,
#         default=1e-6,
#         help="Weight decay magnitude for the Adam optimizer.",
#     )
#     parser.add_argument(
#         "-ae",
#         "--adam_epsilon",
#         type=float,
#         default=1e-08,
#         help="Epsilon value for the Adam optimizer.",
#     )
#     parser.add_argument(
#         "-ul", "--use_lion", type=bool, default=False, help="use Lion optimizer"
#     )
#     parser.add_argument(
#         "-ic",
#         "--mask_channels",
#         type=int,
#         default=1,
#         help="input channels for training (default: 3)",
#     )
#     parser.add_argument(
#         "-c",
#         "--input_img_channels",
#         type=int,
#         default=3,
#         help="output channels for training (default: 3)",
#     )
#     parser.add_argument(
#         "-is",
#         "--image_size",
#         type=int,
#         default=128,
#         help="input image size (default: 128)",
#     )
#     parser.add_argument(
#         "-dd", "--data_path", default="./data", help="directory of input image"
#     )
#     parser.add_argument("-d", "--dim", type=int, default=64, help="dim (default: 64)")
#     parser.add_argument(
#         "-e",
#         "--epochs",
#         type=int,
#         default=10000,
#         help="number of epochs (default: 10000)",
#     )
#     parser.add_argument(
#         "-bs",
#         "--batch_size",
#         type=int,
#         default=4,
#         help="batch size to train on (default: 8)",
#     )
#     parser.add_argument(
#         "--timesteps",
#         type=int,
#         default=1000,
#         help="number of timesteps (default: 1000)",
#     )
#     parser.add_argument("-ds", "--dataset", default="generic", help="Dataset to use")
#     parser.add_argument(
#         "--save_every", type=int, default=100, help="save_every n epochs (default: 100)"
#     )
#     parser.add_argument(
#         "--load_model_from", default=None, help="path to pt file to load from"
#     )
#     return parser.parse_args()

def main():
    
    # Start of main..
    start = time.time()
    # args = parse_args()

    ## Get the Configs..
    config = get_args_config()

    wandb_tracker = None

    ## Set all Seeds..
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    

    checkpoint_dir = os.path.join(config.TRAIN.OUTPUT_DIR, config.MODEL.NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.TRAIN.ACCELERATOR_CONFIG.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=config.TRAIN.ACCELERATOR_CONFIG.MIXED_PRECISION,
        log_with=config.TRAIN.ACCELERATOR_CONFIG.REPORT_TO,
        logging_dir='logs',
    )
    if accelerator.is_main_process:
        # accelerator.init_trackers("experiment", config=vars(args))
        # accelerator.init_trackers("unet_vanced", config=vars(args))
        accelerator.init_trackers(config.DATASET.DATASET_LOADER,init_kwargs={"wandb":{'resume':'allow', 'id':config.MODEL.NAME}})#, config=vars(args))
        wandb_tracker = accelerator.get_tracker('wandb')
        # wandb_tracker.generate_id()

    if accelerator.device == 'cuda':
        torch.cuda.manual_seed(config.SEED)

        if not config.TRAIN.DETERMINISTIC:
            backends.cudnn.benchmark = True
            backends.cudnn.deterministic = False
        else:
            backends.cudnn.benchmark = False
            backends.cudnn.deterministic = True

    
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
    
    # model = UNet_3Plus_attn_FF()
    
    # Dynamically Load the Model, based on the Config..
    # 'from model.UNet_3Plus_attn_FF import UNet_3Plus_attn_FF'
    module = __import__(name=f'model.{config.MODEL.MODEL_TYPE}',fromlist = [f'{config.MODEL.MODEL_TYPE}'])
    model = getattr(module, f'{config.MODEL.MODEL_TYPE}')(**vars(config.MODEL.MODEL_CONFIG))

    # print(f'Checkpoint 1 :{time.time()-start}')
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
    
    train_data_loader, val_data_generator, test_data_generator = load_data(config)
    
    # print(f'Checkpoint 2 :{time.time()-start}')
    # print(type(train_data_generator),type(val_data_generator), type(test_data_generator))
    
    # training_generator = tqdm(data_loader, total=int(len(data_loader)))
    # if args.scale_lr:
    #     args.learning_rate = (
    #         args.learning_rate
    #         * args.gradient_accumulation_steps
    #         * args.batch_size
    #         * accelerator.num_processes
    #     )
    
    
    # Initialize optimizer
    optimizer = None
    if config.TRAIN.OPTIM_NAME == 'SGD':
        print("Using SGD Optimizer.")
        optimizer = SGD(
            model.parameters(), 
            lr=float(str(config.TRAIN.OPTIM_CONFIG.learning_rate)), 
            momentum=0.9, 
            weight_decay=0.0001
        )
        
    elif config.TRAIN.OPTIM_NAME == 'Lion':
        print("Using LION Optimizer.")
        optimizer = Lion(
            model.parameters(),
            lr=float(str(config.TRAIN.OPTIM_CONFIG.learning_rate)),
            betas=(float(str(config.TRAIN.OPTIM_CONFIG.adam_beta1)), float(str(config.TRAIN.OPTIM_CONFIG.adam_beta2))),
            weight_decay=float(str(config.TRAIN.OPTIM_CONFIG.adam_weight_decay)),
        )
    elif config.TRAIN.OPTIM_NAME == 'AdamW':
        print("Using Adam Optimizer.")
        optimizer = AdamW(
            model.parameters(),
            lr=float(str(config.TRAIN.OPTIM_CONFIG.learning_rate)),
            betas=(float(str(config.TRAIN.OPTIM_CONFIG.adam_beta1)), float(str(config.TRAIN.OPTIM_CONFIG.adam_beta2))),
            weight_decay=float(str(config.TRAIN.OPTIM_CONFIG.adam_weight_decay)),
            eps=float(str(config.TRAIN.OPTIM_CONFIG.adam_epsilon)),
        )

    
    # TRAIN MODEL
    counter = 0
    
    model, optimizer, train_data_loader = accelerator.prepare(
        model, optimizer, train_data_loader
    )

    # print(type(train_data_loader))

    print("using device: ", accelerator.device)
    model = model.to(accelerator.device)

    # print(f'Checkpoint 3 :{time.time()-start}')
    cur_epoch = 0
    try:
        if config.MODEL.PRETRAIN_CKPT is not None:
            print(f"Loading model from previous checkpoint: {config.MODEL.PRETRAIN_CKPT}_cur.pt'")
            load_path = str(f'output/{config.MODEL.PRETRAIN_CKPT}/{config.MODEL.PRETRAIN_CKPT}_cur.pt')

            if (os.path.exists(load_path) == False):
                raise Exception("Checkpoint does not exist, conitnuing..")
            
            # save_dict = torch.load(load_path)
            # print(wandb.restore(load_path))
            # print(json.load(open(load_path, encoding='utf-8')))
            with open(load_path,'rb') as f:
                save_dict = torch.load(f,pickle_module=pickle)
                # print(save_dict)
            wandb.restore(load_path)
            model.load_state_dict(save_dict["model_state_dict"])
            optimizer.load_state_dict(save_dict["optimizer_state_dict"])
            cur_epoch = save_dict['epoch'] + 1
            cur_loss = save_dict['loss']

            accelerator.print(f"Loaded from {load_path}")
    except Exception as e:
        cur_epoch = 0
        print(f'Failed: {str(e)}')
        # return
        # print("Did not find config.MODEL.PRETRAIN_CKPT, Moving on..")

    # print(f'Checkpoint 4 :{time.time()-start}')
    # printing number of params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params of {config.MODEL.NAME} model: {pytorch_total_params:,}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params of {config.MODEL.NAME} model: {pytorch_total_params:,}")

    # take smaller steps as we reach minimum
    scheduler = ReduceLROnPlateau(optimizer, "min")  # TODO: uncomment this

    min_epoch_dice_loss = 1

    img_snapshot = None
    mask_snapshot = None

    accelerator.log({"Dice loss": 0})
    accelerator.log({"Train IOU": 0})

    accelerator.log({"Mean DICE on train set": 0})
    accelerator.log({"Mean DICE on validation set": 0})
    accelerator.log({"Mean Dice loss (training epoch)": 0})

    # run_step = 0
    # Iterate across training loop
    for epoch in tqdm(range(cur_epoch, config.TRAIN.MAX_EPOCHS)):
        running_loss = 0.0
        losses = []
        train_iou = []
        # print(f'Checkpoint 5 :{time.time()-start}')
        print(f"Epoch {epoch + 1}/{config.TRAIN.MAX_EPOCHS}")
        train_batch_counter = 0
        for img, mask in tqdm(train_data_loader):
            # print(f'Checkpoint 6 :{time.time()-start}')
            with accelerator.accumulate(model):
                
                # print(img.size(), mask.size())
                img = img.to(accelerator.device)
                mask = mask.to(accelerator.device)
                
                # Take an image snapshot somewhere in the middle of the complete loader..
                
                if (train_batch_counter < len(train_data_loader)/2):
                    img_snapshot = img
                    mask_snapshot = mask

                outputs = model(img)
                out_cut = np.copy(outputs.data.cpu().numpy())
                # print(out_cut.shape, out_cut.min(), out_cut.max())

                out_cut[np.nonzero(out_cut < 0.5)] = 0.0
                out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

                train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
                train_iou.append(train_dice)

                loss = train_loss(outputs, mask)
                losses.append(loss.item())

                # loss = diffusion(mask, img)
                # if train_batch_counter % config.WANDB.LOG_STEP_INTERVAL == 0:
                # accelerator.log({"Dice loss": loss.item()})  # Log loss to wandb
                # accelerator.log({"Train IOU": train_dice})  # Log IOU to wandb
                    
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                train_batch_counter+=1
            
            torch.cuda.empty_cache()

        running_loss += loss.item() * img.size(0)
        counter += 1
        epoch_loss = running_loss / len(train_data_loader)
        print("Epoch Loss : {:.4f}".format(epoch_loss))

        # print(f'Checkpoint 7 :{time.time()-start}')

        mean_dice_train = np.array(train_iou).mean()
        

        val_mean_iou = None
        if val_data_generator is not None:
            val_mean_iou = compute_iou(
                model, val_data_generator, device=accelerator.device
            )
        
        # print(f'Checkpoint 8 :{time.time()-start}')
        mean_loss = np.array(losses).mean()
        
        # print(f'Checkpoint 9 :{time.time()-start}')
        print(
            "Mean loss on train:",
            mean_loss,
            "\nMean DICE on train:",
            mean_dice_train,
            "\nMean DICE on validation:",
            val_mean_iou,
        )

        scheduler.step(mean_loss)  # TODO: uncomment this

        accelerator.log({"Mean DICE on train set": float(mean_dice_train), "Mean DICE on validation set": float(val_mean_iou), "Mean Dice loss (training epoch)": float(mean_loss)})
        # accelerator.log({"Mean DICE on validation set": float(val_mean_iou)})
        # accelerator.log({"Mean Dice loss (training epoch)": float(mean_loss)})

        # print(f'Checkpoint 10 :{time.time()-start}')
        # INFERENCE
        # SAVE BEST MODEL..
        if epoch_loss < min_epoch_dice_loss:
            if epoch % config.TRAIN.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join( config.TRAIN.OUTPUT_DIR, config.MODEL.NAME ,f"{config.MODEL.NAME}_best.pt" )
                
                with open(checkpoint_path,'wb') as f:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            # 'loss': loss.cpu().detach().numpy(),
                            "loss": mean_loss,
                        },f,pickle_module=pickle)
                
                wandb.save(checkpoint_path)

            min_epoch_dice_loss = epoch_loss

        # print(f'Checkpoint 11 :{time.time()-start}')

        # BACKUP SAVE (current model..)
        if epoch % config.TRAIN.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join( config.TRAIN.OUTPUT_DIR, config.MODEL.NAME ,f"{config.MODEL.NAME}_cur.pt" )
            
            with open(checkpoint_path,'wb') as f:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # 'loss': loss.cpu().detach().numpy(),
                        "loss": mean_loss,
                    },f,pickle_module=pickle)
            
            wandb.save(checkpoint_path)

        # print(f'Checkpoint 12 :{time.time()-start}')

        # optimizer_to(optimizer, accelerator.device)
        if epoch % config.WANDB.SAVE_OUTPUT_EPOCH_INTERVAL == 0:
            pred = model(img_snapshot)
            pred_out_cut = np.copy(pred.cpu().detach().numpy())
            pred_out_cut[np.nonzero(pred_out_cut < 0.5)] = 0.0
            pred_out_cut[np.nonzero(pred_out_cut >= 0.5)] = 1.0

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    # save just one image per batch
                    wandb.log(
                        {
                            "img-mask-pred": [
                                wandb.Image(img_snapshot[0, 0, :, :]),
                                wandb.Image(mask_snapshot[0, 0, :, :]),
                                wandb.Image(pred_out_cut[0, 0, :, :]),
                            ]
                        }
                    )

        # print(f'Checkpoint 13 :{time.time()-start}')

if __name__ == "__main__":
    main()
