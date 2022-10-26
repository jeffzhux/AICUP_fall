# init
seed = 2022
amp = True

#data
data_root = './data'
num_workers = 8
num_classes = 37
data = dict(
    collate_ID = dict(
        type = 'MixupCollate',
        num_classes = num_classes
    ),
    collate_OOD = dict(
        type = 'OtherMixupCollate',
        num_classes = num_classes
    ),
    train_ID = dict(
        root=f'{data_root}/ID/train',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='baseOnImageNet'
        ),
    ),
    train_OOD = dict(
        root=f'{data_root}/OOD/train',
        start_class = 32,
        end_class = num_classes,
        type = 'Others_ImageFolder',
        transform = dict(
            type='baseOnImageNet'
        )
    ),
    valid_ID = dict(
        root=f'{data_root}/ID/valid',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='base'
        )
    ),
    valid_OOD = dict(
        root=f'{data_root}/OOD/valid',
        start_class = 32,
        end_class = num_classes,
        type = 'Others_ImageFolder',
        transform = dict(
            type='baseOnImageNet'
        )
    ),
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
    type = 'InOutLoss'
)
#train
epochs = 20
batch_size = 128    #256 cause out of memory & broke pipe error

# optimizer
lr = 0.002
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
work_dir = './ood_experiment/ood_experiment/efficient'
port = 10001
resume = None # (路徑) 從中斷的地方開始 train
load = './experiment/efficient/20221024_104633/epoch_50.pth' # (路徑) 載入訓練好的模型 test