# init
seed = 2022

# test
batch_size = 32

#data
data_root = './data'
num_workers = 8
num_classes = 33

data = dict(
    collate = dict(
        type = 'ClipTestTimeCollateFunction',
        context_length = 2
    ),
    test = dict(
        # root=f'{data_root}/ID_720/valid',
        # loc_file_path = './data/ID/tag_locCoor.csv',
        root=f'{data_root}/Test_720',
        loc_file_path = './data/Test_720/tag_loccoor_public_private.csv',
        type = 'Clip_TestTime_ImageFolder',
        transform = dict(
            type='base',
            size = (320, 320)
        ),
        training_transform= [
            dict(type='fixTest', resize = (384, 384), cropsize = (320, 320)),
        ],
        times=1
    ),
    idx_to_classes = {
        0: 'asparagus', 1: 'bambooshoots', 2: 'betel', 3: 'broccoli', 4: 'cauliflower', 5: 'chinesecabbage', 6: 'chinesechives',
        7: 'custardapple', 8: 'grape', 9: 'greenhouse', 10: 'greenonion', 11: 'kale', 12: 'lemon', 13: 'lettuce', 14: 'litchi',
        15: 'longan', 16: 'loofah', 17: 'mango', 18: 'onion', 19: 'papaya', 20: 'passionfruit', 21: 'pear', 22: 'pennisetum',
        23: 'redbeans', 24: 'roseapple', 25: 'sesbania', 26: 'soybeans', 27: 'sunhemp', 28: 'sweetpotato', 29: 'taro', 30: 'tea',
        31: 'waterbamboo', 32: 'others'
    }
)

# model
model_ema = dict(
    status = True,
    steps=2,
    decay=0.99998
)

# model
backbone_s = dict(type = 'efficientnet_v2_s',num_classes = num_classes)
backbone_m = dict(type = 'efficientnet_v2_m',num_classes = num_classes)
models = [
    dict(type="ClipNet", backbone=backbone_s),
    dict(type="ClipNet", backbone=backbone_s),
    dict(type="ClipNet", backbone=backbone_s),
    dict(type="ClipNet", backbone=backbone_m),
]


#log & save
draw = False
save_pred = False
output_file_name = 'submission'
work_dir = './test_experiment/final_two_ema'
load = [
    './experiment/efficientV2S_Progressing4/base1_2/20221206_092628/epoch_100.pth',
    './experiment/efficientV2S_semi/20221209_091753/epoch_100.pth',
    './experiment/efficientV2S_Progressing5/base1_2/20221210_203426/epoch_80.pth', #noise student
    './experiment/efficientV2M_Progressing1/base1_2/20221212_144039/epoch_80.pth'
]
port = 10001

