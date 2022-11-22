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
    train = dict(
        root=f'{data_root}/ID/train',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='base',
        ),
        base_transform = dict(
            type='base',
            size = (352, 352)
        ),
        num_of_trans = 0
    ),
    test = dict(
        root=f'{data_root}/ID/valid',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='base',
        ),
        base_transform = dict(
            type='base',
            size = (224, 224)
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
batch_size = 32

#log & save
output_file_name = None#'submission'
work_dir = './test_experiment/efficient'
load = './experiment/efficientV2S/20221120_201850/epoch_100.pth' 
# load = './experiment/efficientV2S/20221118_230640/epoch_100.pth' # 8742 224,224
# load = './experiment/efficientV2S/20221119_220253/epoch_100.pth' # 8922 356,356
port = 10001

