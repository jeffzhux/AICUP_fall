# init
seed = 2022

#data
num_workers = 4
num_classes = 32
data_root = './data'
data = dict(
    collate = dict(
        type = 'MixupCollate',
        num_classes = num_classes
    ),
    
    id = dict(
        root=f'{data_root}/ID/train',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='baseOnImageNet'
        ),
    ),
    ood = dict(
        root=f'{data_root}/OOD/train',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='baseOnImageNet'
        ),
    ),
    vaild = dict(
        root=f'{data_root}/ID/valid',
        type = 'AICUP_ImageFolder',
        transform = dict(
            type='base'
        )
    )
)
# model
model = dict(
    type="EfficientNet_Base",
    backbone = dict(
        type = 'efficientnet_b0',
        num_classes = num_classes 
    )
)
# train
epochs = 10
batch_size = 64
# loss
train_loss = dict(
    type = 'OELoss',#'EnergyLoss'
)
valid_loss = dict(
    type = 'CrossEntropyLoss'
)

# optimizer
lr = 0.001
optimizer = dict(
    type='SGD',
    lr = lr,
    momentum = 0.9,
    weight_decay = 5e-4

)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs * 112,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    #start_step=0,
    warmup_steps=0, # 100
    #warmup_from=1e-6
    min = 1e-6 # 收斂到最小值
)

# log & save
log_interval = 20
save_interval = 5
work_dir = './odd_experiment/energy'
resume = None # (路徑) 從中斷的地方開始 train
load = './experiment/efficient/20221013_130305/epoch_350.pth'
port = 10001