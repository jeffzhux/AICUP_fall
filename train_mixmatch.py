
import os
import platform
import argparse
import numpy as np
from datasets.collates.build import build_collate
from utils.config import Config
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from utils.util import AverageMeter, TrackMeter, accuracy, adjust_learning_rate, format_time, set_seed, set_weight_decay
from utils.build import build_logger
from datasets.sampler import build_sampler
from datasets.build import build_dataset
from models.build import build_model, build_ema_model
from losses.build import build_loss
from datasets.transforms import build_transform
from optimizers.build import build_optimizer
from datasets.transforms.augmentations import RandomMixupCutMix

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

def load_weights(ckpt_path, model, model_ema, optimizer, scaler, resume=True) -> int:
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    if resume:
        # load model & optimizer
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler'])

        if model_ema is not None:
            model_ema.load_state_dict(checkpoint['model_ema_state'])
        start_epoch = checkpoint['epoch'] + 1
        print("Loaded. (epoch {})".format(checkpoint['epoch']))
    else:
        # load model & optimizer
        model.load_state_dict(checkpoint['model_state'])

        if model_ema is not None:
            model_ema.load_state_dict(checkpoint['model_ema_state'])
        start_epoch = 1

    return start_epoch

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def train(model, model_ema, labeled_dataloader, unlabeled_dataloader, augmentation, criterion, optimizer, epoch, scaler, cfg, logger=None, writer=None):
    model.train() # 開啟batch normalization 和 dropout
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    labeled_train_iter = iter(labeled_dataloader)
    unlabeled_train_iter = iter(unlabeled_dataloader)
    num_iter = len(labeled_dataloader)
    iter_end = time.time()
    epoch_end = time.time()
    for idx in range(num_iter):
        try:
            imgs, labels, loc, text = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_dataloader)
            imgs, labels, loc, text = labeled_train_iter.next()

        try:
            imgs_u, imgs_u2, loc_u, text_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_dataloader)
            imgs_u, imgs_u2, loc_u, text_u = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - iter_end)

        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        loc = loc.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)

        imgs_u = imgs.cuda(non_blocking=True)
        imgs_u2 = imgs.cuda(non_blocking=True)
        loc_u = loc.cuda(non_blocking=True)
        text_u = text.cuda(non_blocking=True)
        
        with torch.no_grad():
            outputs_u = model(imgs_u, loc_u, text_u)
            outputs_u2 = model(imgs_u2, loc_u, text_u)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1 / 0.5) # T = 0.5 sharpens
            labels_u = pt / pt.sum(dim=1, keepdim=True)
            labels_u = labels_u.detach()
        
        # mixup label & unlabel
        labels = labels = F.one_hot(labels, labels_u.size(-1))
        all_img = torch.cat([imgs, imgs_u, imgs_u2], dim=0)
        all_labels = torch.cat([labels, labels_u, labels_u], dim=0)
        all_loc = torch.cat([loc, loc_u, loc_u], dim=0)
        all_text = torch.cat([text, text_u, text_u], dim=0)
        all_img, all_labels, lam, index = augmentation(all_img, all_labels)
        
        # compute output
        batch_size = imgs.size(0)
        all_img = list(torch.split(all_img, batch_size))
        all_img = interleave(all_img, batch_size)
        with autocast(enabled=scaler is not None):
            logits = [model(all_img[0], loc, text)]
            for input_img in all_img[1:]:
                logits.append(model(input_img, loc_u, text_u))

            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)
            loss = criterion(logits_x, all_labels[:batch_size], logits_u, all_labels[batch_size:], epoch)

        losses.update(loss.item(), batch_size)

        # accurate
        acc1, acc5 = accuracy(torch.cat(logits, dim=0), all_labels, topk=(1,5))
        top1.update(acc1.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if model_ema and idx % cfg.model_ema.steps == 0:
            model_ema.update_parameters(model)
            if epoch < cfg.lr_cfg.warmup_steps:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)
        
        # measure elapsed time
        batch_time.update(time.time() - iter_end)
        iter_end = time.time()

        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:  # cfg.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch[epoch][idx/iter] [{epoch}][{idx+1:>4d}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},  '
                        f'batch_time: {batch_time.avg:.3f},  '
                        f'lr: {lr:.5f},  '
                        f'loss(loss avg): {loss:.3f}({losses.avg:.3f}),  '
                        f'train_Acc@1: {top1.avg:.3f}  '
            )
        
    if logger is not None: 
        now = time.time()
        epoch_time = format_time(now - epoch_end)
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time},  '
                    f'train_loss: {losses.avg:.3f},  '
                    f'train_Acc@1: {top1.avg:.3f}  ')
    
    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/lr', lr, epoch)
        writer.add_scalar('Train/loss', losses.avg, epoch)
        writer.add_scalar('Train/acc@1', top1.avg, epoch)

def valid(model, dataloader, criterion, optimizer, epoch, cfg, logger, writer):
    model.eval() # 開啟batch normalization 和 dropout

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    test_meter = TrackMeter()
    end = time.time()

    with torch.no_grad():
        for idx, (images, targets, loc, text) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            loc = loc.cuda(non_blocking=True)
            text = text.cuda(non_blocking=True)

            batch_size = targets.shape[0]

            # forward
            logits = model(images, loc, text)
            loss = F.cross_entropy(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1,5))

            # update metric
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

    epoch_time = format_time(time.time() - end)

    ''' save best
    if top1.avg > test_meter.max_val:
        model_path = os.path.join(cfg.work_dir, f'best_{cfg.cfgname}.pth')
        state_dict={
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'acc': top1.avg,
            'epoch':epoch
        }
        torch.save(state_dict, model_path)
    '''

    if logger is not None:
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'valid_loss: {losses.avg:.3f}, '
                    f'valid_Acc@1: {top1.avg:.3f}, '
                    f'valid_Acc@5: {top5.avg:.3f}')

    if writer is not None:
        writer.add_scalar('Valid/loss', losses.avg, epoch)
        writer.add_scalar('Valid/acc@1', top1.avg, epoch)
    

def main_worker(rank, world_size, cfg):
    print(f'==> start rank: {rank}')

    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(rank)
    set_seed(cfg.seed+rank, cuda_deterministic=False)

    print(f'System : {platform.system()}')
    if platform.system() == 'Windows':
        dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)
    else: # Linux
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    # build logger, writer
    logger, writer = None, None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))
        logger = build_logger(cfg.work_dir, 'train')

    cfg.bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', cfg.bsz_gpu)
    
    # build dataset
    train_set =  build_dataset(cfg.data.train)
    train_collate = build_collate(cfg.data.collate)
    train_sampler = build_sampler(train_set, cfg.data.sampler)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.bsz_gpu,
        num_workers=cfg.num_workers,
        collate_fn = train_collate,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    unlabeled_set = build_dataset(cfg.data.unlabel)
    unlabeled_collate = build_collate(cfg.data.unlabel_collate)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_set,
        batch_size=cfg.bsz_gpu,
        num_workers=cfg.num_workers,
        collate_fn = unlabeled_collate,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    valid_set = build_dataset(cfg.data.vaild)
    valid_collate = build_collate(cfg.data.collate)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=cfg.bsz_gpu,
        num_workers=cfg.num_workers,
        collate_fn = valid_collate,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    #Augmentation
    augmentation = RandomMixupCutMix(**cfg.data.augmentation)

    # build model
    model = build_model(cfg.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    # build criterion
    criterion = build_loss(cfg.loss).cuda()
    # build optimizer
    # parameters = set_weight_decay(model, cfg.weight_decay)
    optimizer = build_optimizer(cfg.optimizer, model.parameters())
    # fp16 or fp32
    scaler = GradScaler() if cfg.amp else None

    #如果網路當中有不需要backward的find_unused_parameters 要設為 True
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], find_unused_parameters=False)
    model_without_ddp = model.module

    model_ema = build_ema_model(model_without_ddp, cfg)

    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, model_without_ddp, model_ema, optimizer, scaler, resume=True)
    elif cfg.load:
        start_epoch = load_weights(cfg.load, model_without_ddp, model_ema, optimizer, scaler, resume=False)
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer, epoch)

        # train; all processes
        train(model, model_ema, train_loader, unlabeled_loader, augmentation, criterion, optimizer, epoch, scaler, cfg, logger, writer)
        
        if model_ema:
            valid(model_ema, valid_loader, criterion, optimizer, epoch, cfg, logger, writer)
        else:
            valid(model, valid_loader, criterion, optimizer, epoch, cfg, logger, writer)
        
        # save ckpt; master process
        if rank == 0 and epoch % cfg.save_interval == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state': optimizer.state_dict(),
                'model_state': model_without_ddp.state_dict(),
                'epoch': epoch
            }
            if scaler:
                state_dict['scaler'] = scaler.state_dict()
            if model_ema:
                state_dict['model_ema_state'] = model_ema.state_dict()
            torch.save(state_dict, model_path)

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