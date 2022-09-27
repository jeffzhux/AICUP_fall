# init
seed = 2022

#data
data_root = './data'
num_workers = 2
data = dict(
    train = dict(
        root=f'{data_root}/train',
        transform = dict(
            type='base'
        )
    ),
    vaild = dict(
        root=f'{data_root}/vaild',
        transform = dict(
            type='base'
        )
    ),
    test = dict(
        root=f'{data_root}/test',
        transform = dict(
            type='base'
        )
    ),
)

# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_b0',
        weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1',
        num_classes = 33
    )
    
)

# loss
loss = dict(
    type = 'CrossEntropyLoss'

)
#train
epochs = 1
batch_size = 32

# optimizer
lr = 0.5
optimizer = dict(
    type='SGD',
    lr = lr,
    momentum = 0.9,
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
log_interval = 20
save_interval = 100
work_dir = './experiment/efficient'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
#load = None # (路徑) 載入訓練好的模型 test