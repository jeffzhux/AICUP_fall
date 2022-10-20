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

def split_class_other(pred, group_list):
    def get_group_slice(group_list):
        start = 0
        slice_group = []
        for i in group_list:
            end = start + len(i) + 1
            slice_group.append([start, end])
            start = end
        return slice_group

    slice_group = get_group_slice(group_list)

    classes_group_softmax = []
    others_group_softmax = []
    for slice in slice_group:
        group_logit = pred[:, slice[0]:slice[1]]

        group_softmax = F.softmax(group_logit, dim=1) 
        classes_group_softmax.append(group_softmax[:,:-1])# disregard others category
        # others_group_softmax.append(torch.special.entr(group_softmax[:,:-1]))
        others_group_softmax.append(group_softmax[:,-1:]) # others categorys


    classes_group_softmax = torch.cat(classes_group_softmax, dim=1)
    others_group_softmax = torch.cat(others_group_softmax, dim=1)

    return classes_group_softmax, others_group_softmax


@torch.no_grad()
def iterate_data(model, dataloader, tta, cfg):
    
    pred_class = []
    pred_other = []
    target = []
    
    for idx, (images, labels) in enumerate(dataloader):
        images = images.float().cuda()
        labels = labels.cuda()

        # forward
        logits = model(images)
        logits, other_logits = split_class_other(logits, cfg.group_list)

        logits = tta(logits)
        other_logits = tta(other_logits)

        pred_class.append(logits)
        pred_other.append(other_logits)
        target.append(labels)
        # print('20221014 要記得拿掉 break')
        # break ########### 記得要拿掉

    pred_class = torch.cat(pred_class)
    pred_other = torch.cat(pred_other)
    target = torch.cat(target)

    return pred_class, pred_other, target

@torch.no_grad()
def run_eval(model, id_test_loader, ood_test_loader, tta, cfg):
    in_pred_class, in_pred_other, in_target = iterate_data(model, id_test_loader, tta, cfg)
    # out_pred_class, out_pred_other, out_target = iterate_data(model, ood_test_loader, tta, cfg)
    
    # out_target += 32

    # in_num, out_num = in_pred_class.size(0), out_pred_class.size(0)
    # pred_class = torch.cat((in_pred_class, out_pred_class))
    # pred_other = torch.cat((in_pred_other, out_pred_other))
    # target = torch.cat((in_target, out_target))

    # ood_score, _ = torch.max(-pred_other, dim=1)

    # pred_class =
    # in_score = mos(pred_other)
       
    acc1, acc5 = accuracy(in_pred_class, in_target, topk=(1, 5))
    print(acc1)
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
    # out_pred_class, out_pred_other, out_target = iterate_data(model, ood_test_loader, tta, cfg)
    # in_pred_class, in_pred_other, in_target = iterate_data(model, id_test_loader, tta, cfg)

    # id_score, _ = torch.min(in_pred_other, dim=1)
    
    # id_score = torch.special.entr(in_pred_other).sum(-1)
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