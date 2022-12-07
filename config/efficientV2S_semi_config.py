# init
seed = 1022
amp = False

#train
epochs = 1#100
batch_size = 16#256

#data
data_root = './data'
num_workers = 8
num_classes = 33
data = dict(
    augmentation = dict(
        num_classes = num_classes,
        mixup_alpha = 0.75,
        cutmix_alpha = 0.75,
        is_mixmatch = True
    ),
    collate = dict(
        type = 'ClipCollateFunction',
        context_length = 2
    ),
    unlabel_collate = dict(
        type = 'ClipUnlabelCollateFunction',
        context_length = 2
    ),
    sampler = dict(
        type='RASampler',
        shuffle = True,
        repetitions = 4
    ),
    train = dict(
        root=f'{data_root}/ID_720/train',
        type = 'Clip_ImageFolder',
        loc_file_path = './data/ID/tag_locCoor.csv',
        transform = dict(
            type='baseOnTrivialAugment',
            size = (128, 128),
            lighting = 0.1
        )
    ),
    unlabel = dict(
        root=f'{data_root}/Test_720',
        type = 'ClipUnlabel_ImageFolder',
        loc_file_path = './data/Test/tag_loccoor_public.csv',
        transform = dict(
            type='baseOnTrivialAugment',
            size = (128, 128),
        )
    ),
    vaild = dict(
        root=f'{data_root}/ID_720/valid',
        type = 'Clip_ImageFolder',
        loc_file_path = './data/ID/tag_locCoor.csv',
        transform = dict(
            type='base',
            size = (128, 128)
        )
    )
)

# model
model_ema = dict(
    status = True,
    steps=1,
    decay=0.99998
)

model = dict(
    type="ClipNet",
    backbone = dict(
        type = 'efficientnet_v2_s',
        # weights = 'EfficientNet_V2_S_Weights.IMAGENET1K_V1',
        num_classes = num_classes,
    )
    
)

# loss
loss = dict(
    type = 'MixmatchLoss',
    label_smoothing = 0.1,
    rampup_length = epochs
)


# optimizer
lr = 0.03
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
    warmup_steps=5, # 100
    warmup_from=lr * 0.1
)


#log & save
log_interval = 100
save_interval = 20
work_dir = './experiment/efficientV2S_semi'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
load = None
# load = './experiment/efficientV2S_Progressing3/base1_1/20221127_234539/epoch_100.pth' # (路徑) 載入訓練好的模型 test

