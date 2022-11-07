# init
seed = 2022

#data
data_root = './data/ID'
num_workers = 8
num_classes = 33
test_time_augmentation = dict(
    num_of_trans = 0,
    merge_mode = 'mean',
    sharpen = 0.5 # weight of original image
)
groups = [
    ['asparagus', 'onion', 'others', 'greenhouse', 'chinesecabbage', 'roseapple', 'passionfruit'],
    ['sesbania', 'lemon','litchi', 'chinesechives', 'pennisetum', 'longan', 'cauliflower', 'lettuce', 'loofah', 'custardapple', 'pear'],
    ['greenonion', 'papaya', 'mango', 'betel', 'bambooshoots', 'taro', 'waterbamboo', 'grape', 'kale', 'sweetpotato', 'broccoli', 'redbeans', 'soybeans', 'sunhemp', 'tea']
]
groups_range = [
    (0, 7),
    (7, 18),
    (18, 33)
]
data = dict(
    collate = dict(
        type=None
    ),
    test = dict(
        root=f'{data_root}/valid',
        type = 'Group_ImageFolder',
        transform = dict(
            type='base'
        ),
        groups = groups,
        groups_range = groups_range,
    )
    
)

# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_v2_s',
        num_classes = num_classes + len(groups_range)
    )
)

# test
batch_size = 16

#log & save
work_dir = './test_experiment/efficient'
load = './experiment/efficient_sam_group/20221103_153449/epoch_100.pth'
port = 10001

