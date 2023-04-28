# Delete WANDB Runs..

# import wandb
# api = wandb.Api()
# run = api.run("gsamarth97/ISICDataset/ISIC_UNet_img128_ep150")
# run.delete()

## Clear Cache..
import os
import shutil
import platform
print(f'Platform: {platform.platform()}')

# (Clean PC from all the caches..)

# Do this when your project is finished..

## Cleaning WandB Checkpoint Cache..

for (root,dirs,files) in os.walk('C:/Users/gsama/AppData/Local/Temp', topdown=True):
        # print (root)
        for dir in dirs:
            if 'wandb' in dir:
                print(f'Clearing: {root}/{dir}')
                shutil.rmtree(f'{root}/{dir}')
        # print (files)