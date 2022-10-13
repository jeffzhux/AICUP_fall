import os
import random
import shutil
random.seed(2022)
train_path = './data/train'
valid_path = './data/valid'
train, valid = 0.9, 0.1

for dir in os.listdir(train_path):
    sub_files = os.listdir(os.path.join(train_path, dir))

    if not os.path.exists(os.path.join(valid_path, dir)):
        os.makedirs(os.path.join(valid_path, dir))


    random.shuffle(sub_files)
    len_sub_files = len(sub_files)

    len_train = int(len_sub_files * train)
    len_valid = len_sub_files - len_train

    X_valid = sub_files[len_train: len_train+len_valid]


    for file in X_valid:

        source = rf'{train_path}/{dir}/{file}'
        target = rf'{valid_path}/{dir}/{file}'

        # break
        shutil.move(source, target)

    print(f'{dir} total : {len_sub_files}')
    print(f'{dir} train : {len_train}')
    print(f'{dir} valid : {len_valid}')
    print('-----------------------------------------------')
