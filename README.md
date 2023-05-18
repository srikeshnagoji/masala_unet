

# MASALA U-NET

## Steps before each commit:
```bash
pip list --format=freeze > requirements.txt
```

## Setup:
### Python == 3.9.13
```bash
conda create -p ./venv python==3.9.13 -y
conda activate ./venv
pip install requirements.txt

# Finally, Run this for CUDA Enabled Torch..

pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
pip install torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
```


## Commands (New):

```bashbash
NOTE (WINDOWS) : RUN VSCODE IN ADMIN MODE!!
```

### Config: unet_3plus_attn_FF_img128_ep150
```bash
accelerate launch train.py --cfg configs/unet_3plus_attn_FF_img128_ep150.yaml
```

### Config: swin_tiny_patch4_window7_img224_ep150
```bash
accelerate launch train.py --cfg configs/swin_tiny_patch4_window7_img224_ep150.yaml
```

### Config: BrainMRI_unet_3plus_attn_FF_img128_ep150
```bash
accelerate launch train.py --cfg configs/BrainMRI_unet_3plus_attn_FF_img128_ep150.yaml
```

### Config: ISIC_unet_3plus_attn_FF_img128_ep150
```bash
accelerate launch train.py --cfg configs/ISIC_unet_3plus_attn_FF_img128_ep150.yaml
```

### Config: ISIC_SwinUNet_patch4_window7_img224_ep150
```bash
accelerate launch train.py --cfg configs/ISIC_SwinUNet_patch4_window7_img224_ep150.yaml
```

### Test Commands:
```bash
accelerate launch test.py --cfg configs/ISIC_UNet_img128_ep150.yaml
accelerate launch test.py --cfg configs/ISIC_SwinUNet_img128_ep150.yaml
```


### Sam's Competetors for SwinUnet:
```bash
accelerate launch train.py --cfg configs/ISIC_UNet_img128_ep150_2.yaml
accelerate launch train.py --cfg configs/ISIC_AttentionUNet_img128_ep150.yaml
accelerate launch train.py --cfg configs/ISIC_InceptionUNet_img128_ep150.yaml
accelerate launch train.py --cfg configs/ISIC_UNet3Plus_img128_ep150.yaml
```


### Commands (deprecated):
```bash
accelerate launch train.py --dataset ISIC --mask_channels=1 --input_img_channels=3 --image_size=64 --data_path='/Users/srikeshnagoji/Documents/PythonWorkSpace/jupyter_lab_workspace/PES/final_thesis/dataset/ISIC_skin'  --dim=64 --epochs=6 --save_every 5
```
```bash
accelerate launch train.py --dataset ISIC --mask_channels=1 --input_img_channels=3 --image_size=64 --data_path='/Users/srikeshnagoji/Documents/PythonWorkSpace/jupyter_lab_workspace/PES/final_thesis/dataset/ISIC_skin'  --dim=64 --epochs=6 --save_every 5 --use_lion True
```
```bash
accelerate launch train.py --dataset brain --mask_channels=1 --input_img_channels=3 --image_size=64 --data_path='/Users/srikeshnagoji/Documents/PythonWorkSpace/jupyter_lab_workspace/PES/CAPSTONE/kaggle_3m'  --dim=64 --epochs=151 --save_every 5
```

## Latest Changes (Sam):
```
- Added working support for AttentionUNet, InceptionUNet, UNet3+.
- Added Test flow
- Config added for:
    - Mixed Loss (CE Loss+DICE) or just DICE Loss
    
- Tweaked SwinUNet Parameters
- Optimized WANDB checkpointing models forever.
- Reference for changes: ISIC_SwinUNet_img128_ep150.yaml
```