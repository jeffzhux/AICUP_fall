# init
seed = 2022

#model
backbone = dict(
    type="Efficient",
    depth="b0",
    num_classes = 33

)

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

#train
epochs = 1
batch_size = 254

#log & save
work_dir = './experiment/efficient'
port = 10001