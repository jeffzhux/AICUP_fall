import zipfile
from io import BytesIO
import os
import glob
from PIL import Image
import pandas as pd
import math
data_path = './data_public_train/others-004.zip'
center_file = './training/tag_locCoor.csv'
TARGET_W = 512 # 目標寬度 +1 
TARGET_H = 512 # 目標長度 +1

df = pd.read_csv(center_file, encoding='ANSI')


zippedImgs = zipfile.ZipFile(data_path)
 
folder_path = f'./data/others'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
for file_in_zip in zippedImgs.namelist()[1:]:
    
    file_name = os.path.split(file_in_zip)[1]

    data = zippedImgs.read(file_in_zip)
    dataEnc = BytesIO(data)
    img = Image.open(dataEnc)
    W, H = img.size
    num_of_split = max(H, W) / min(H, W)
    if num_of_split < 1.5: 
        temp_w, temp_h = min(W,H), min(W,H)

        (bias_x, bias_y) = list(df.loc[df['Img'] == file_name][['target_x', 'target_y']].itertuples(index=False, name=None))[0]
        center_x, center_y = math.ceil(W/2) + bias_x, math.ceil(H/2) - bias_y
        left = center_x - int(temp_w/2)
        upper = center_y - int(temp_h/2)
        right = center_x + int(temp_w/2) + 1
        lower = center_y + int(temp_h/2) + 1

        if left < 0:
            right += (0 - left)
            right = min(right, W)
            left = 0
        if right > W:
            left -= (right-W)
            left = max(0, left)
            right = W
        if upper < 0:
            lower += (0 - upper)
            lower = min(lower, H)
            upper = 0
        if lower > H:
            upper -= (lower - H)
            upper = max(0, upper)
            lower = H
        img = img.crop((left, upper, right, lower))
    
        img = img.resize((TARGET_W, TARGET_H))
        img.save(f'{folder_path}/{file_name}')
    else:
        
        temp_w, temp_h = min(W,H), min(W,H)
        left, upper = 0, 0
        right, lower = temp_w, temp_h
        for i in range(round(num_of_split)):

            image = img.crop((left, upper, right, lower))
            
            image = image.resize((TARGET_W, TARGET_H))
            image.save(f"{folder_path}/{file_name.replace('.',f'_{i}.')}")

            if W == min(H, W):
                lower = min(lower + temp_w, H)
                upper = lower - temp_h
            else:
                right = min(right + temp_h, W)
                left = right - temp_w
            

        
