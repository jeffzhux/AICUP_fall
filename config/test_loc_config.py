# init
seed = 2022

# test
batch_size = 32
draw = False

#data
data_root = './data'
num_workers = 8
num_classes = 33
test_time_augmentation = dict(
    num_of_trans = 0,
    merge_mode = 'mean',
    sharpen = 0.5 # weight of original image
)

data = dict(
    collate = dict(
        type = 'locCollateFunction',
    ),
    test = dict(
        root=f'{data_root}/ID/valid',
        type = 'loc_ImageFolder',
        transform = dict(
            type='base',
            size = (480, 480)
        ),
    )
    
)

# model
model_ema = dict(
    status = True,
    steps=32,
    decay=0.99998
)

# model
model = dict(
    type="LocClipNet",
    backbone = dict(
        type = 'efficientnet_v2_s',
        dropout_rate = 0.1,
        num_classes = num_classes,
        batch_size = batch_size,
    )
)



#log & save
output_file_name = None#'submission'
work_dir = './test_experiment/efficient_loc'
load = './experiment/efficientV2S_Progressing3/base1_3/20221201_102338/epoch_40.pth'
# load = './experiment/efficientV2S_Progressing3/base1_2/20221128_102223/epoch_100.pth'
port = 10001

