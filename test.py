import os
import platform
import argparse
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from datasets.build import build_dataset
from datasets.collates.build import build_collate
from models.build import build_model
from utils.config import Config
from utils.util import AverageMeter, TrackMeter, accuracy, adjust_learning_rate, format_time, set_seed
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

def load_weights(
    ckpt_path: str,
    model: nn.Module) -> None:

    # load checkpoint 
    print(f"==> Loading Checkpoint {ckpt_path}")
    assert os.path.isfile(ckpt_path), 'file is not exist'
    ckpt = torch.load(ckpt_path, map_location='cuda')

    model.load_state_dict(ckpt['model_state'])

@torch.no_grad()
def test(model, dataloader, cfg, logger):
    tta = TestTimeAugmentation(cfg.test_time_augmentation)
    pred = []
    target = []

    for idx, (images, labels) in enumerate(dataloader):
        images = images.float().cuda()
        labels = labels.cuda()

        # forward
        logits = model(images)
        logits = tta(logits)
        
        pred.append(logits)
        target.append(labels)

    pred = torch.cat(pred)
    target = torch.cat(target)

    acc1, acc5 = accuracy(pred, target, topk=(1, 5))
    acc1, acc5 = acc1.item(), acc5.item()
    if logger is not None:
        logger.info(f'Acc@1: {acc1:.3f}, '
                    f'Acc@5: {acc5:.3f}')


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

    logger, writer = None, None
    if rank == 0:
        # writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))
        logger = build_logger(cfg.work_dir, 'test')

    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    test_set = build_dataset(cfg.data.test)
    test_collate = build_collate(cfg.data.collate)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = bsz_gpu,
        num_workers = cfg.num_workers,
        collate_fn = test_collate,
        pin_memory = True,
        drop_last = False
    )

    model = build_model(cfg.model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    
    if cfg.load:
        load_weights(cfg.load, model)

    cudnn.benchmark = True
    
    print(f"==> Start testing ....")
    model.eval()

    test(model, test_loader, cfg, logger)
    
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