import os
import platform
import argparse
import time
from sklearn.metrics import f1_score

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
def run_eval(model, id_test_loader, ood_test_loader, tta, cfg):
    
    in_pred_class, in_target = iterate_data(model, id_test_loader, tta, cfg)
    out_pred_class, out_target = iterate_data(model, ood_test_loader, tta, cfg)
    
    pred_class = torch.cat((in_pred_class, out_pred_class))
    pred_class = F.softmax(pred_class, dim=-1)
    score = torch.special.entr(pred_class).sum(-1)
    score = torch.where(score > 2.7, 1, 0).view(-1,1)
    pred_class = torch.cat((pred_class, score), dim=-1)

    out_target[:] = 32
    target = torch.cat((in_target, out_target))

    metrix = Metric(pred_class.size(-1))
    metrix.update(pred_class, target)
    recall = metrix.recall('none')
    precision = metrix.precision('none')
    f1_score = metrix.f1_score('none')
    acc = metrix.accuracy('none')
    
    print(recall)
    print(precision)
    print(f1_score)
    print(acc)
    # in_num, out_num = in_pred_class.size(0), out_pred_class.size(0)
    # pred_class = torch.cat((in_pred_class, out_pred_class))
    # pred_other = torch.cat((in_pred_other, out_pred_other))
    # target = torch.cat((in_target, out_target))

    # ood_score, _ = torch.max(-pred_other, dim=1)

    # pred_class =
    # in_score = mos(pred_other)
       
    # acc1, acc5 = accuracy(in_pred_class, in_target, topk=(1, 5))
    # print(acc1)
    # metrix = Metric(cfg.num_classes)
    # metrix.update(pred_class, target)
    # recall = metrix.recall('none')
    # precision = metrix.precision('none')

    # # print(metrix.weighted_precision())
    # acc1, acc5 = acc1.item(), acc5.item()
    # if logger is not None:
    #     logger.info(f'Acc@1: {acc1:.3f}, '
    #                 f'Acc@5: {acc5:.3f}, '
    #                 f'recall: {recall.mean():.3f}, '
    #                 f'precision: {precision.mean():.3f}')


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

    logger = None
    if rank == 0:
        logger = build_logger(cfg.work_dir, 'test')

    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    id_test_set = build_dataset(cfg.data.id_test)
    id_test_collate = build_collate(cfg.data.collate)
    id_test_loader = torch.utils.data.DataLoader(
        id_test_set,
        batch_size = bsz_gpu,
        num_workers = cfg.num_workers,
        shuffle = False,
        collate_fn = id_test_collate,
        pin_memory = True,
        drop_last = False
    )

    ood_test_set = build_dataset(cfg.data.ood_test)
    ood_test_collate = build_collate(cfg.data.collate)
    ood_test_loader = torch.utils.data.DataLoader(
        ood_test_set,
        batch_size = bsz_gpu,
        num_workers = cfg.num_workers,
        shuffle = False,
        collate_fn = ood_test_collate,
        pin_memory = True,
        drop_last = False
    )

    model = build_model(cfg.model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    tta = TestTimeAugmentation(cfg.test_time_augmentation)
    if cfg.load:
        load_weights(cfg.load, model)

    cudnn.benchmark = True
    
    print(f"==> Start testing ....")
    model.eval()
    run_eval(model, id_test_loader, ood_test_loader, tta, cfg)
    # out_pred_class, out_target = iterate_data(model, ood_test_loader, tta, cfg)
    # in_pred_class, in_target = iterate_data(model, id_test_loader, tta, cfg)

    # id_score = torch.special.entr(F.softmax(in_pred_class, dim=-1)).sum(-1)
    
    # torch.save(id_score, './id_score.pt')
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