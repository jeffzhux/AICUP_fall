# init
seed = 1022
amp = False

#data
data_root = './data/ID'
num_workers = 8
num_classes = 33
data = dict(
    collate = dict(
        type = 'RandomMixupCutMixCollate',
        num_classes = num_classes,
        mixup_alpha=0.1
    ),
    sampler = dict(
        type='RASampler',
        shuffle = True,
        repetitions = 4
    ),
    train = dict(
        root=f'{data_root}/valid',
        type = 'CCrop_ImageFolder',
        transform = dict(
            type='baseOnContrasive',
            size = (128, 128),
            alpha = 0.6,
            lighting = 0.1
        )
    ),
    cc_train = dict(
        root=f'{data_root}/valid',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='base',
            size = (128, 128)
        )
    ),
    vaild = dict(
        root=f'{data_root}/valid',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='base',
            size = (128, 128)
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
        dropout_rate = 0.1,
        num_classes = num_classes 
    )
    
)

# loss
loss = dict(
    type = 'CrossEntropyLoss',
    label_smoothing = 0.1
)

# boxes
warmup_epochs = 5#10
loc_interval = 20#20
box_thresh = 0.2

#train
epochs = 100
batch_size = 512

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
log_interval = 200
save_interval = 20
work_dir = './experiment/efficientV2S_cc'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
load = None # (路徑) 載入訓練好的模型 test

