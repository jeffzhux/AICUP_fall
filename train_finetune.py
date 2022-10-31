
import os
import platform
import argparse
from datasets.collates.build import build_collate
from utils.config import Config
import time
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter
from utils.util import AverageMeter, TrackMeter, accuracy, adjust_learning_rate, format_time, set_seed
from utils.build import build_logger
from datasets.build import build_dataset
from models.build import build_model
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

def load_weights(ckpt_path: str, model: nn.Module) -> None:

    # load checkpoint 
    print(f"==> Fine tune Loading Checkpoint {ckpt_path}")
    assert os.path.isfile(ckpt_path), 'file is not exist'
    ckpt = torch.load(ckpt_path, map_location='cuda')

    model.load_state_dict(ckpt['model_state'])
    

def train(model, dataloader, criterion, optimizer, epoch, cfg, logger=None, writer=None):
    model.train() # 開啟batch normalization 和 dropout
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    num_iter = len(dataloader)
    iter_end = time.time()
    epoch_end = time.time()
    for idx, (imgs, labels) in enumerate(dataloader):
        
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        batch_size = imgs.size(0)

        # measure data loading time
        data_time.update(time.time() - iter_end)

        # compute output
        logits= model(imgs)
        loss = criterion(logits, labels)

        losses.update(loss.item(), batch_size)

        # accurate
        acc1, acc5 = accuracy(logits, labels, topk=(1,5))
        top1.update(acc1.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # smp first forward-backward pass
        with model.no_sync():
            loss.backward()
        optimizer.first_step(zero_grad=True)
        # smp second forward-backward pass
        criterion(model(imgs), labels).backward()
        optimizer.second_step(zero_grad=True)

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

    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)
    
    # build dataset

    train_set =  build_dataset(cfg.data.train)
    train_collate = build_collate(cfg.data.collate)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        collate_fn = train_collate,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    valid_set = build_dataset(cfg.data.vaild)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # build model
    model = build_model(cfg.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    #如果網路當中有不需要backward的find_unused_parameters 要設為 True
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], find_unused_parameters=False)
    
    # build criterion
    criterion = build_loss(cfg.loss).cuda()
    # build optimizer
    optimizer = build_optimizer(cfg.optimizer, model.parameters())

    start_epoch = 1
    if cfg.load:
        load_weights(cfg.load, model)

    cudnn.benchmark = True
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer.base_optimizer, epoch)

        # train; all processes
        train(model, train_loader, criterion, optimizer, epoch, cfg, logger, writer)
        
        valid(model, valid_loader, criterion, optimizer, epoch, cfg, logger, writer)

        # save ckpt; master process
        if rank == 0 and epoch % cfg.save_interval == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state': optimizer.state_dict(),
                'model_state': model.state_dict(),
                'epoch': epoch
            }
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