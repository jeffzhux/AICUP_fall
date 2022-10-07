# init
seed = 2022

#data
data_root = './data'
num_workers = 8
data = dict(
    # train = dict(
    #     root=f'./cifar10_data/train',
    #     type = 'CIFAR10',
    #     download=False,
    #     train = True,
    #     transform = dict(
    #         type='cifar10_train'
    #     )
    # ),
    # vaild = dict(
    #     root=f'./cifar10_data/train',
    #     type = 'CIFAR10',
    #     download=False,
    #     train = False,
    #     transform = dict(
    #         type='cifar10_valid'
    #     )
    # ),
    collate = dict(
        type = 'None'
    ),
    train = dict(
        root=f'{data_root}/train',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='baseWithAim'
        )
    ),
    vaild = dict(
        root=f'{data_root}/valid',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='base'
        )
    ),
    test = dict(
        root=f'{data_root}/test',
        type = 'AICUP_ImageFolder',
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
epochs = 200
batch_size = 128

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
log_interval = 20
save_interval = 5
work_dir = './experiment/efficient'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
#load = None # (路徑) 載入訓練好的模型 test