# init
seed = 2022

#data
data_root = './data/ID'
num_workers = 8
num_classes = 32
data = dict(
    collate = dict(
        type = 'OtherMixupCollate',
        num_classes = num_classes
    ),
    train = dict(
        root=f'{data_root}/train',
        type = 'ImageFolderWithOther',
        transform = dict(
            type='baseOnImageNet'
        ),
        target_transform = dict(
            type='otherOneHotTransform'
        )
    ),
    vaild = dict(
        root=f'{data_root}/valid',
        type = 'ImageFolderWithOther',
        transform = dict(
            type='base'
        ),
        target_transform = dict(
            type='otherOneHotTransform'
        )
    )
)

# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_b0',
        weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1',
        num_classes = num_classes 
    )
    
)

# loss
loss = dict(
    type = 'CrossEntropyLoss'
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
save_interval = 5
work_dir = './experiment/ood_experiment/efficient'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
#load = None # (路徑) 載入訓練好的模型 test