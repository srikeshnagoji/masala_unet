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

from metrics.diceMetrics import dice_coef_metric, compute_iou, DiceLoss
from accelerate import Accelerator
import wandb

from torch import backends

import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensor

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from util.helper import get_args_config, optimizer_to


def main():
    # args = parse_args()

    ## Get the Configs..
    config = get_args_config()

    wandb_tracker = None

    ## Set all Seeds..
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    

    # checkpoint_dir = os.path.join(config.TRAIN.OUTPUT_DIR, config.MODEL.NAME)
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    # logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.TEST.ACCELERATOR_CONFIG.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=config.TEST.ACCELERATOR_CONFIG.MIXED_PRECISION,
        log_with=config.TEST.ACCELERATOR_CONFIG.REPORT_TO,
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
    'from model.UNet_3Plus_attn_FF import UNet_3Plus_attn_FF'
    module = __import__(name=f'model.{config.MODEL.MODEL_TYPE}',fromlist = [f'{config.MODEL.MODEL_TYPE}'])
    model = getattr(module, f'{config.MODEL.MODEL_TYPE}')(**vars(config.MODEL.MODEL_CONFIG))

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
    
    train_data_generator, val_data_generator, test_data_generator = load_data(config)
    
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
    elif config.TRAIN.OPTIM_NAME == 'Adam':
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
        model, optimizer, train_data_generator
    )
    print("using device: ", accelerator.device)
    model = model.to(accelerator.device)

    cur_epoch = 0
    try:
        if config.MODEL.PRETRAIN_CKPT is not None:
            print(f"Loading model from checkpoint: {config.MODEL.PRETRAIN_CKPT}_best.pt'")
            load_path = str(f'output/{config.MODEL.PRETRAIN_CKPT}/{config.MODEL.PRETRAIN_CKPT}_best.pt')
            # save_dict = torch.load(load_path)
            save_dict = torch.load(wandb.restore(load_path))
            model.load_state_dict(save_dict["model_state_dict"])
            optimizer.load_state_dict(save_dict["optimizer_state_dict"])
            cur_epoch = save_dict['epoch'] + 1
            cur_loss = save_dict['loss']

            accelerator.print(f"Loaded from {load_path}")
    except Exception as e:
        cur_epoch = 0
        print("Did not find config.MODEL.PRETRAIN_CKPT, Moving on..")

    # printing number of params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params of {model_name} model: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params of {model_name} model: {pytorch_total_params}")

    # take smaller steps as we reach minimum
    scheduler = ReduceLROnPlateau(optimizer, "min")  # TODO: uncomment this

    min_epoch_dice_loss = 1

    # Iterate across training loop
    for epoch in tqdm(range(cur_epoch, config.TRAIN.MAX_EPOCHS)):
        running_loss = 0.0
        losses = []
        train_iou = []

        print("Epoch {}/{}".format(epoch + 1, config.TRAIN.MAX_EPOCHS))
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
        accelerator.log({"Epoch Loss": epoch_loss})

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

        # INFERENCE
        # optimizer_to(optimizer, "cpu")
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
                            "loss": loss,
                        },f)
                
                wandb.save(checkpoint_path)

            min_epoch_dice_loss = epoch_loss

            # optimizer_to(optimizer, accelerator.device)
        if epoch % config.WANDB.SAVE_OUTPUT_INTERVAL == 0:
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