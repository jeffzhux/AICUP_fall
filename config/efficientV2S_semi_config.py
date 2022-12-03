# init
seed = 1022
amp = False
do_semi = True

#train
epochs = 1#100
batch_size = 16#256

#data
data_root = './data'
num_workers = 8
num_classes = 33
data = dict(
    collate = dict(
        type = 'locCollate',
        num_classes = num_classes,
        mixup_alpha=0.2
    ),
    sampler = dict(
        type='RASampler',
        shuffle = True,
        repetitions = 4
    ),
    train = dict(
        root=f'{data_root}/ID/train',
        type = 'loc_ImageFolder',
        loc_file_path = './data/ID/tag_locCoor.csv',
        transform = dict(
            type='baseOnTrivialAugment',
            size = (224, 224),
            lighting = 0.1
        )
    ),
    unlabedled = dict(
        root=f'{data_root}/Test',
        type = 'locwithPath_ImageFolder',
        loc_file_path = './data/Test/tag_loccoor_public.csv',
        transform = dict(
            type='base',
            size = (224, 224),
        )
    ),
    vaild = dict(
        root=f'{data_root}/ID/valid',
        type = 'loc_ImageFolder',
        loc_file_path = './data/ID/tag_locCoor.csv',
        transform = dict(
            type='base',
            size = (224, 224)
        )
    )
)

# model
model_ema = dict(
    status = True,
    steps=32,
    decay=0.99998
)

model = dict(
    type="LocClipNet",
    backbone = dict(
        type = 'efficientnet_v2_s',
        # weights = 'EfficientNet_V2_S_Weights.IMAGENET1K_V1',
        dropout_rate = 0.1,
        num_classes = num_classes,
        batch_size = batch_size
    )
    
)

# loss
loss = dict(
    type = 'CrossEntropyLoss',
    label_smoothing = 0.1
)


# optimizer
lr = 0.005
weight_decay = 1e-4
optimizer = dict(
    type = 'SGD',
    lr = lr,
    momentum = 0.9,
    weight_decay = weight_decay,
)

lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    #start_step=0,
    warmup_steps=0, # 100
    # warmup_from=lr * 0.1
)


#log & save
log_interval = 100
save_interval = 20
work_dir = './experiment/efficientV2S_semi'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
load = './experiment/efficientV2S_Progressing3/base1_1/20221127_234539/epoch_100.pth' # (路徑) 載入訓練好的模型 test

