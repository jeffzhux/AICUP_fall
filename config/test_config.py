# init
seed = 2022

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
        type = 'TestTimeCollate',
    ),
    test = dict(
        root=f'{data_root}/ID/valid',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='base',
        ),
        base_transform = dict(
            type='base',
            size = (448,448)
        ),
        num_of_trans = test_time_augmentation['num_of_trans']
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
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_v2_s',
        num_classes = num_classes 
    )
)

# test
batch_size = 16

#log & save
work_dir = './test_experiment/efficient'
load = './experiment/efficientV2S/20221112_000349/epoch_25.pth'
# load = './experiment/efficient_sam/20221107_132000/epoch_100.pth'

port = 10001

