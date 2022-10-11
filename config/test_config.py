# init
seed = 2022

#data
data_root = './data'
num_workers = 8
num_classes = 33
data = dict(
    collate = dict(
        type = 'TestTimeCollate',
    ),
    test = dict(
        root=f'{data_root}/test',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type=None
        )
    ),
)

# model
model = dict(
    type="ResNet_Base"
)

# test
batch_size = 2

#log & save
work_dir = './test_experiment/resnet'
load = './experiment/resnet/20220926_125813/best_resnet_config.pth'
port = 10001

