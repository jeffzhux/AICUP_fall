# init
seed = 2022

#data
data_root = './data/ID'
group_list = [
    ['asparagus', 'onion', 'greenhouse', 'chinesecabbage', 'roseapple', 'passionfruit'],
    ['sesbania', 'lemon', 'litchi', 'chinesechives', 'pennisetum', 'longan', 'cauliflower', 'lettuce', 'loofah', 'custardapple', 'pear'],
    ['greenonion', 'papaya', 'mango', 'betel', 'bambooshoots', 'taro', 'waterbamboo', 'grape', 'kale', 'sweetpotato', 'broccoli', 'redbeans', 'soybeans', 'sunhemp', 'tea']
]
group_num = 3
num_workers = 8
num_classes = 32
data = dict(
    collate = dict(
        type = 'GroupMixupCollate',
        num_classes = num_classes
    ),
    train = dict(
        root=f'{data_root}/train',
        type = 'ImageFolderWithGroup',
        transform = dict(
            type='baseOnImageNet'
        ),
        group_list = group_list
    ),
    vaild = dict(
        root=f'{data_root}/valid',
        type = 'ImageFolderWithGroup',
        transform = dict(
            type='base'
        ),
        group_list = group_list
    )
)

# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_b0',
        weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1',
        num_classes = num_classes + group_num
    )
    
)

# loss
loss = dict(
    type = 'GroupMixUpLoss',
    group_list = group_list
)
#train
epochs = 1
batch_size = 16

# optimizer
lr = 0.001
optimizer = dict(
    type='Adam',
    lr = lr,
    # momentum = 0.9,
    weight_decay = 1e-4

)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    #start_step=0,
    warmup_steps=0, # 100
    #warmup_from=1e-6
)


#log & save
log_interval = 100
save_interval = 50
work_dir = './ood_experiment/efficient'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
#load = None # (路徑) 載入訓練好的模型 test