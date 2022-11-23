
import os
import platform
import argparse
from datasets.collates.build import build_collate
from utils.config import Config
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch


from torch.utils.tensorboard import SummaryWriter
from utils.util import AverageMeter, TrackMeter, accuracy, adjust_learning_rate, format_time, set_seed, set_weight_decay
from utils.build import build_logger
from datasets.sampler import build_sampler
from datasets.build import build_dataset
from datasets.dataloader import ConcatDataLoader
from models.build import build_model, build_ema_model
from losses.build import build_loss
from optimizers.build import build_optimizer

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

def train(model, model_ema, dataloader, criterion, optimizer, epoch, scaler, cfg, logger=None, writer=None):
    model.train() # 開啟batch normalization 和 dropout
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    t_top1 = AverageMeter()

    num_iter = len(dataloader)
    iter_end = time.time()
    epoch_end = time.time()
    for idx, ((s_imgs, s_labels), (t_imgs, t_labels)) in enumerate(dataloader):
        
        s_imgs = s_imgs.cuda(non_blocking=True)
        s_labels = s_labels.cuda(non_blocking=True)

        t_imgs = t_imgs.cuda(non_blocking=True)
        t_labels = t_labels.cuda(non_blocking=True)

        batch_size = s_imgs.size(0)

        # measure data loading time
        data_time.update(time.time() - iter_end)

        # compute output
        with autocast(enabled=scaler is not None):
            s_logits= model(s_imgs, reverse=False)
            t_logits = model(t_imgs, reverse=True)

            loss = criterion(s_logits, t_logits, s_labels, t_labels)

        losses.update(loss.item(), batch_size)

        # accurate
        acc1, acc5 = accuracy(s_logits, s_labels, topk=(1,5))
        t_acc1, acc5 = accuracy(t_logits, t_labels, topk=(1,5))
        top1.update(acc1.item(), batch_size)
        t_top1.update(t_acc1.item(), batch_size)

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
        writer.add_scalar('OSDA/lr', lr, epoch)
        writer.add_scalar('OSDA/loss', losses.avg, epoch)
        writer.add_scalar('OSDA/OSacc@1', top1.avg, epoch)
        writer.add_scalar('OSDA/OS*acc@1', t_top1.avg, epoch)

def valid(model, dataloader, criterion, optimizer, epoch, cfg, logger, writer):
    model.eval() # 開啟batch normalization 和 dropout

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    test_meter = TrackMeter()
    end = time.time()

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            batch_size = targets.shape[0]

            # forward
            logits = model(images)
            loss = criterion(logits, targets)
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
    source_set =  build_dataset(cfg.data.source)
    source_collate = build_collate(cfg.data.collate)
    source_sampler = build_sampler(source_set, cfg.data.sampler)
    source_loader = torch.utils.data.DataLoader(
        source_set,
        batch_size=cfg.bsz_gpu,
        num_workers=cfg.num_workers,
        collate_fn = source_collate,
        pin_memory=True,
        sampler=source_sampler,
        drop_last=True
    )
    target_set =  build_dataset(cfg.data.target)
    target_collate = build_collate(cfg.data.collate)
    target_sampler = build_sampler(target_set, cfg.data.sampler)
    target_loader = torch.utils.data.DataLoader(
        target_set,
        batch_size=cfg.bsz_gpu,
        num_workers=cfg.num_workers,
        collate_fn = target_collate,
        pin_memory=True,
        sampler=target_sampler,
        drop_last=True
    )
    train_loader = ConcatDataLoader(source_loader, target_loader)

    valid_set = build_dataset(cfg.data.vaild)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=cfg.bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # build model
    model = build_model(cfg.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    # build criterion
    train_criterion = build_loss(cfg.train_loss).cuda()
    valid_criterion = build_loss(cfg.valid_loss).cuda()
    # build optimizer
    # parameters = set_weight_decay(model, cfg.weight_decay)
    optimizer = build_optimizer(cfg.optimizer, model.parameters())
    # fp16 or fp32
    scaler = GradScaler() if cfg.amp else None

    #如果網路當中有不需要backward的find_unused_parameters 要設為 True
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[cfg.local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False)

    model_without_ddp = model.module

    model_ema = build_ema_model(model_without_ddp, cfg)

    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, model_without_ddp, model_ema, optimizer, scaler, resume=True)
    elif cfg.load:
        start_epoch = load_weights(cfg.load, model_without_ddp, model_ema, optimizer, scaler, resume=False)
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        source_sampler.set_epoch(epoch)
        target_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer, epoch)

        # train; all processes
        train(model, model_ema, train_loader, train_criterion, optimizer, epoch, scaler, cfg, logger, writer)
        
        if model_ema:
            valid(model_ema, valid_loader, valid_criterion, optimizer, epoch, cfg, logger, writer)
        else:
            valid(model, valid_loader, valid_criterion, optimizer, epoch, cfg, logger, writer)
        
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