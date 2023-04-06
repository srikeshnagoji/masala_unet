# MASALA U-NET

#### Command:

accelerate launch driver.py --dataset ISIC --mask_channels=1 --input_img_channels=3 --image_size=64 --data_path='/Users/srikeshnagoji/Documents/PythonWorkSpace/jupyter_lab_workspace/PES/final_thesis/dataset/ISIC_skin'  --dim=64 --epochs=6 --save_every 5


accelerate launch driver.py --dataset ISIC --mask_channels=1 --input_img_channels=3 --image_size=64 --data_path='/Users/srikeshnagoji/Documents/PythonWorkSpace/jupyter_lab_workspace/PES/final_thesis/dataset/ISIC_skin'  --dim=64 --epochs=6 --save_every 5 --use_lion True


accelerate launch driver.py --dataset brain --mask_channels=1 --input_img_channels=3 --image_size=64 --data_path='/Users/srikeshnagoji/Documents/PythonWorkSpace/jupyter_lab_workspace/PES/CAPSTONE/kaggle_3m'  --dim=64 --epochs=151 --save_every 5