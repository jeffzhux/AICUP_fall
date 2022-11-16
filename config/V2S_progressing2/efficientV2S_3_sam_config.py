# init
seed = 3022
amp = False

#data
data_root = './data/ID'
num_workers = 8
num_classes = 33
data = dict(
    collate = dict(
        type = 'RandomMixupCutMixCollate',
        num_classes = num_classes,
        mixup_alpha=0.3
    ),
    sampler = dict(
        type='RASampler',
        shuffle = True,
        repetitions = 4
    ),
    train = dict(
        root=f'{data_root}/train',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='baseOnTrivialAugment',
            size = (300, 300),
            lighting = 0.1
        )
    ),
    vaild = dict(
        root=f'{data_root}/valid',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='base',
            size = (300, 300)
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
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_v2_s',
        weights = 'EfficientNet_V2_S_Weights.IMAGENET1K_V1',
        dropout_rate = 0.4,
        num_classes = num_classes 
    )
    
)

# loss
loss = dict(
    type = 'CrossEntropyLoss',
    label_smoothing = 0.1
)
#train
epochs = 25#100
batch_size = 128#256

# optimizer
lr = 0.0025
weight_decay = 1e-04
optimizer = dict(
    type = 'SAM',
    rho = 2.0,
    adaptive = True,
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
log_interval = 200
save_interval = 5
work_dir = './experiment/efficientV2S_Progressing2/base1_3'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
load = './experiment/efficientV2S_Progressing2/base1_2/20221114_233151/epoch_100.pth' # (路徑) 載入訓練好的模型 test

