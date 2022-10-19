# init
seed = 2022

#data
data_root = './data'
num_workers = 8
num_classes = 32
test_time_augmentation = dict(
    num_of_trans = 0,
    merge_mode = 'mean',
    sharpen = 0.5 # weight of original image
)
out_of_distribution = dict(
    type='EnergyOOD',
    #mode='softmax', #softmax entropy
    temperature=1
)
data = dict(
    collate = dict(
        type = 'TestTimeCollate',
    ),
    test = dict(
        root=f'{data_root}/OOD/valid',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='baseOnImageNet'
        ),
        num_of_trans = test_time_augmentation['num_of_trans']
    ),
    # ood_test = dict(
    #     root=f'{data_root}/OOD/valid',
    #     type = 'TestTimeAICUP_DataSet',
    #     transform = dict(
    #         type='baseOnImageNet'
    #     ),
    #     num_of_trans = test_time_augmentation['num_of_trans']
    # ),
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
load = './ood_experiment/20221017_131232/epoch_10.pth'
port = 10001

