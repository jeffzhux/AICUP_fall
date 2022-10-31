# init
seed = 2022

#data
data_root = './data/ID'
num_workers = 8
num_classes = 33
data = dict(
    collate = dict(
        type = 'MixupCollate',
        num_classes = num_classes
    ),
    train = dict(
        root=f'{data_root}/train',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='baseOnImageNet'
        )
    ),
    vaild = dict(
        root=f'{data_root}/valid',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='Base'
        )
    )
)

# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_v2_s',
        weights = 'EfficientNet_V2_S_Weights.IMAGENET1K_V1',
        num_classes = num_classes 
    )
    
)

# loss
loss = dict(
    type = 'CrossEntropyLoss'
)
#train
epochs = 1
batch_size = 16#256

# optimizer
lr = 0.01
optimizer = dict(
    type = 'SAM',
    rho = 2.0,
    adaptive = True,
    lr = lr,
    momentum = 0.9,
    weight_decay = 5e-4,
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
save_interval = 50
work_dir = './experiment/efficient'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
load = None # (路徑) 載入訓練好的模型 test

