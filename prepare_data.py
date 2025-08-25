import os
from shutil import copyfile
from tqdm import tqdm
import cv2
import numpy as np

src_dir = '/media/adithya/home_data/SR/Dataset_v12/valid/clean_denoised'


sr_dirs = ['/media/adithya/home_data/SR/Dataset_v12/valid/sr_0_denoised',
           '/media/adithya/home_data/SR/Dataset_v12/valid/sr_1_denoised',
           '/media/adithya/home_data/SR/Dataset_v12/valid/sr_3_denoised',
           '/media/adithya/home_data/SR/Dataset_v12/valid/sr_4_denoised'
          ]

dst_dir = '/media/adithya/home_data/Deblur/Dataset_v1/val'

os.makedirs(dst_dir, exist_ok=True)

for img_name in tqdm(os.listdir(src_dir)):
    try:
        images = []
        if img_name.endswith('.png'):
            img_path = os.path.join(src_dir, img_name)
            target_image = cv2.imread(img_path)
            for sr_dir, factor in zip(sr_dirs, [-2, -1, 1, 2]):
                copy_img_name = img_name.split('.')[0] + f'_{factor}.png'
                source_image = cv2.imread(os.path.join(sr_dir, img_name))
                save_image = np.hstack((source_image, target_image))
                cv2.imwrite(os.path.join(dst_dir, copy_img_name), save_image)
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        continue