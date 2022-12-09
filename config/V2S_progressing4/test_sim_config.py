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
        type = 'ClipCollateFunction',
        context_length = 2
    ),
    test = dict(
        # root=f'{data_root}/ID_720/valid',
        # loc_file_path = './data/ID/tag_locCoor.csv',
        root=f'{data_root}/Test_720',
        loc_file_path = './data/Test_720/tag_loccoor_public.csv',
        type = 'Clip_ImageFolder',
        transform = dict(
            type='base',
            size = (224, 224)
        ),
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
model = dict(
    type="ClipNet",
    backbone = dict(
        type = 'efficientnet_v2_s',
        num_classes = num_classes,
    )
)



#log & save
output_file_name = 'submission'
work_dir = './test_experiment/efficient_sim'
# load = './experiment/efficientV2S_Progressing4/base1_1/20221205_233405/epoch_100.pth'
# load = './experiment/efficientV2S_Progressing4/base1_2/20221206_092628/epoch_40.pth'
load = './experiment/efficientV2S_semi/20221209_091753/epoch_20.pth'

port = 10001

