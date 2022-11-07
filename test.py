import os
import platform
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from datasets.build import build_dataset
from datasets.collates.build import build_collate
from models.build import build_model
from utils.config import Config
from utils.util import Metric, accuracy, set_seed
from utils.build import build_logger
from utils.test_time_augmentation import TestTimeAugmentation

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    args = parser.parse_args()

    return args

def get_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)

    cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    cfg.work_dir = os.path.join(cfg.work_dir, f'{cfg.timestamp}')

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

def load_weights(ckpt_path: str, model: nn.Module) -> None:

    # load checkpoint 
    print(f"==> Loading Checkpoint {ckpt_path}")
    assert os.path.isfile(ckpt_path), 'file is not exist'
    ckpt = torch.load(ckpt_path, map_location='cuda')

    model.load_state_dict(ckpt['model_state'])

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

    pred_class = torch.cat(pred_class)
    target = torch.cat(target)

    return pred_class, target

@torch.no_grad()
def run_eval(model, test_loader, tta, cfg):
    
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
    print(pred_class.size())
    print(target.size())
    metrix = Metric(pred_class.size(-1))
    metrix.update(pred_class, target)
    recall = metrix.recall('none')
    precision = metrix.precision('none')
    f1_score = metrix.f1_score('none')
    acc = metrix.accuracy('none')
    wp = metrix.weighted_precision(0.5)
    print(acc)
    print(recall)
    print(precision)
    print(f1_score)
    print(wp)

def main_worker(rank, world_size, cfg):
    print(f'==> start rank: {rank}')

    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(rank)

    print(f'System : {platform.system()}')
    if platform.system() == 'Windows':
        dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)
    else: # Linux
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    test_set = build_dataset(cfg.data.test)
    print(test_set.class_to_idx)
    test_collate = build_collate(cfg.data.collate)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = bsz_gpu,
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
    if cfg.load:
        load_weights(cfg.load, model)
    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    print(f"==> Start testing ....")
    model.eval()
    # with torch.inference_mode:
    run_eval(model, test_loader, tta, cfg)

def main():
    args = get_args()
    cfg = get_config(args)

    world_size= torch.cuda.device_count()
    print(f'GPUs on this node: {world_size}')
    cfg.world_size = world_size
    
    log_file = os.path.join(cfg.work_dir, f'{cfg.timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))
if __name__ == '__main__':
    main()