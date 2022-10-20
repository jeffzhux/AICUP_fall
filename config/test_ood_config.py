# init
seed = 2022

#data
data_root = './data'
group_list = [
    ['asparagus', 'onion', 'greenhouse', 'chinesecabbage', 'roseapple', 'passionfruit'],
    ['sesbania', 'lemon', 'litchi', 'chinesechives', 'pennisetum', 'longan', 'cauliflower', 'lettuce', 'loofah', 'custardapple', 'pear'],
    ['greenonion', 'papaya', 'mango', 'betel', 'bambooshoots', 'taro', 'waterbamboo', 'grape', 'kale', 'sweetpotato', 'broccoli', 'redbeans', 'soybeans', 'sunhemp', 'tea']
]
group_num = 3
num_workers = 8
num_classes = 32
test_time_augmentation = dict(
    num_of_trans = 0,
    merge_mode = 'mean',
    sharpen = 0.5 # weight of original image
)

data = dict(
    collate = dict(
        type = 'GroupTestTimeCollate',
    ),
    id_test = dict(
        root=f'{data_root}/ID/valid',
        type = 'TestTimeImageFolderWithGroup',
        transform = dict(
            type='baseOnImageNet'
        ),
        num_of_trans = test_time_augmentation['num_of_trans'],
        group_list = group_list
    ),
    ood_test = dict(
        root=f'{data_root}/OOD/valid',
        type = 'TestTimeImageFolderWithGroup',
        transform = dict(
            type='baseOnImageNet'
        ),
        num_of_trans = test_time_augmentation['num_of_trans'],
        group_list = [['others']]
    ),
)

# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_b0',
        num_classes = num_classes + group_num
    )
)

# test
batch_size = 16

#log & save
work_dir = './test_experiment/efficient'
load = './ood_experiment/efficient/20221019_101634/epoch_50.pth'
port = 10001

