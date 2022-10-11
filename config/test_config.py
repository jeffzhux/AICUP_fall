# init
seed = 2022

#data
data_root = './data'
num_workers = 8
num_classes = 33
test_time_augmentation = dict(
    num_of_trans = 10,
    merge_mode = 'mean',
    sharpen = 0.5 # weight of original image
)
data = dict(
    collate = dict(
        type = 'TestTimeCollate',
    ),
    test = dict(
        root=f'{data_root}/test',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='baseOnImageNet'
        ),
        num_of_trans = test_time_augmentation['num_of_trans']
    ),
)

# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_b0',
        num_classes = num_classes 
    )
)

# test
batch_size = 16

#log & save
work_dir = './test_experiment/efficient'
load = './experiment/efficient/20221007_164732/epoch_50.pth'
port = 10001

