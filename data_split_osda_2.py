import os
import random
import shutil
random.seed(2022)
source_path = './data/OSDA/source'
target_path = './data/OSDA/target'

weight = 2005 / 78546
source, target = 1-weight, weight
all_source = 0
all_target = 0

for dir in os.listdir(source_path):
    sub_files = os.listdir(os.path.join(source_path, dir))

    if not os.path.exists(os.path.join(target_path, dir)):
        os.makedirs(os.path.join(target_path, dir))


    random.shuffle(sub_files)
    len_sub_files = len(sub_files)

    len_source = int(len_sub_files * source)
    len_target = len_sub_files - len_source

    X_valid = sub_files[len_source: len_source+len_target]


    for file in X_valid:
        s = rf'{source_path}/{dir}/{file}'
        t = rf'{target_path}/{dir}/{file}'

        # shutil.move(s, t)
    all_source += len_source
    all_target += len_target
    print(f'{dir} total : {len_sub_files}')
    print(f'{dir} source : {len_source}')
    print(f'{dir} target : {len_target}')
    print('-----------------------------------------------')

print(f'{source_path} : {all_source}')
print(f'{target_path} : {all_target}')
