import os
import platform
import argparse
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from datasets.build import build_dataset
from datasets.collates.build import build_collate
from models.build import build_model, build_ema_model
from utils.config import Config
from utils.util import Metric, accuracy, set_seed
from utils.build import build_logger
from utils.test_time_augmentation import TestTimeAugmentation
from sklearn import metrics
import matplotlib.pyplot as plt
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    args = parser.parse_args()

    return args

def get_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)

    cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # cfg.work_dir = os.path.join(cfg.work_dir, f'{cfg.timestamp}')

    # worker
    cfg.num_workers = min(cfg.num_workers, mp.cpu_count()-2)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None, f'{cfg.cfgname} is not exist'

    # seed
    if not hasattr(cfg, 'seed'):
        cfg.seed = 25
    set_seed(cfg.seed)

    return cfg

def load_weights(ckpt_path: str, model: nn.Module, model_ema: nn.Module) -> None:

    # load checkpoint 
    print(f"==> Loading Checkpoint {ckpt_path}")
    assert os.path.isfile(ckpt_path), 'file is not exist'
    ckpt = torch.load(ckpt_path, map_location='cuda')

    model.load_state_dict(ckpt['model_state'])
    model_ema.load_state_dict(ckpt['model_ema_state'])
@torch.no_grad()
def iterate_data(model, dataloader, tta, cfg):
    
    pred_class = []
    target = []
    
    for idx, (images, labels) in enumerate(dataloader):
        images = images.float().cuda()
        labels = labels.cuda()

        # forward
        logits = model(images)
        logits = tta(logits)

        pred_class.append(logits)
        target.append(labels)
        # break
    pred_class = torch.cat(pred_class)
    target = torch.cat(target)

    return pred_class, target

@torch.no_grad()
def run_eval(model, test_loader, dataset, tta, cfg):
    
    pred_class, target = iterate_data(model, test_loader, tta, cfg)
    
    if hasattr(cfg, 'groups_range'):
        all_group_score = []
        other_group_score = []
        for idx, (start, end) in enumerate(cfg.groups_range):
            other_idx = -len(cfg.groups_range) + idx
            sub_pred = torch.cat((pred_class[:,start:end], pred_class[:,other_idx].view(-1,1)), dim=-1)
            sub_softmax = F.softmax(sub_pred, dim=-1)
            all_group_score.append(sub_softmax[:, :-1]) # except others class
            other_group_score.append(sub_softmax[:,-1:])
        all_group_score = torch.cat(all_group_score, dim=-1)
        other_group_score = torch.cat(other_group_score, dim=-1)

        min_other_score, min_other_idx = torch.min(other_group_score, dim=-1)
        pred_class = all_group_score
        pred_class[:,4] = torch.where(min_other_score > 0.8, 1, pred_class[:,4])

        target = target[:, :-len(cfg.groups_range)]
        target = torch.argmax(target, dim=-1)
    else:
        pred_class = F.softmax(pred_class, dim=-1)

    
    metrix = Metric(pred_class.size(-1))
    metrix.update(pred_class, target)
    confuse_matrix = metrix.confusion_matrix().numpy()
    wp = metrix.weighted_precision(0.7)

    fig, ax = plt.subplots(figsize=(20,20))
    ax.ticklabel_format(style='plain', useOffset=False)
    cm_display = metrics.ConfusionMatrixDisplay(confuse_matrix, display_labels=dataset.class_to_idx.keys())
    cm_display.plot(ax = ax, values_format='.20g', cmap='Blues')
    plt.xticks(rotation=90)
    cm_display.figure_.savefig(os.path.join(cfg.work_dir, f'confuse_metrix.png'))
    print(metrics.classification_report(target.cpu().tolist(), torch.argmax(pred_class, dim=-1).cpu().tolist(), target_names=dataset.class_to_idx.keys()))
    print(f'wp : {wp.item()}')
    
@torch.no_grad()
def run_inference(model, dataloader, dataset, cfg):
    pred_class = []
    idx_to_classes = {
        0: 'asparagus', 1: 'bambooshoots', 2: 'betel', 3: 'broccoli', 4: 'cauliflower', 5: 'chinesecabbage',
        6: 'chinesechives', 7: 'custardapple', 8: 'grape', 9: 'greenhouse', 10: 'greenonion', 11: 'kale',
        12: 'lemon', 13: 'lettuce', 14: 'litchi', 15: 'longan', 16: 'loofah', 17: 'mango', 18: 'onion', 19: 'others',
        20: 'papaya', 21: 'passionfruit', 22: 'pear', 23: 'pennisetum', 24: 'redbeans',25: 'roseapple', 26: 'sesbania',
        27: 'soybeans', 28: 'sunhemp', 29: 'sweetpotato', 30: 'taro', 31: 'tea', 32: 'waterbamboo'}
    
    for idx, (images, _) in enumerate(dataloader):
        images = images.float().cuda()

        # forward
        logits = model(images)
        logits = F.softmax(logits, dim=-1)
        logits = torch.argmax(logits, dim=-1).cpu().tolist()
        
        pred_class.extend(logits)
        
    pred_class = list(map(lambda x: idx_to_classes[x], pred_class))
    file_name_list = list(map(lambda x: x[0].split('\\')[-1], dataset.imgs))
    
    assert len(pred_class)== len(file_name_list), f'pred len is {len(pred_class)}, but file_name_list len is {len(file_name_list)}'
    with open(os.path.join(cfg.work_dir, f'{cfg.output_file_name}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([['filename', 'label']])
        writer.writerows(zip(file_name_list, pred_class))
    return pred_class

def main_worker(rank, world_size, cfg):
    print(f'==> start rank: {rank}')

    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(rank)
    set_seed(cfg.seed+rank, cuda_deterministic=True)

    print(f'System : {platform.system()}')
    if platform.system() == 'Windows':
        dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)
    else: # Linux
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    cfg.bsz_gpu = int(cfg.batch_size / cfg.world_size)
    cfg.epochs = 1
    print('batch_size per gpu:', cfg.bsz_gpu)

    test_set = build_dataset(cfg.data.test)
    print(test_set.class_to_idx)
    test_collate = build_collate(cfg.data.collate)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = cfg.bsz_gpu,
        num_workers = cfg.num_workers,
        shuffle = False,
        collate_fn = test_collate,
        pin_memory = True,
        drop_last = False
    )

    model = build_model(cfg.model)
    model.cuda()
    
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    tta = TestTimeAugmentation(cfg.test_time_augmentation)
    model_without_ddp = model.module
    

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    model_ema = build_ema_model(model_without_ddp, cfg)
    if cfg.load:
        load_weights(cfg.load, model_without_ddp, model_ema)

    print(f"==> Start testing ....")
    model.eval()
    # with torch.inference_mode:
    if cfg.output_file_name is not None:
        run_inference(model, test_loader, test_set, cfg)
    else:
        run_eval(model, test_loader, test_set, tta, cfg)
        
    # run_eval(model_ema, test_loader, tta, cfg)
def main():
    args = get_args()
    cfg = get_config(args)

    world_size= torch.cuda.device_count()
    print(f'GPUs on this node: {world_size}')
    cfg.world_size = world_size

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))
if __name__ == '__main__':
    main()