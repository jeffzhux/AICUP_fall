
import os
import platform
import argparse
from datasets.collates.build import build_collate
from utils.config import Config
import time
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch


from torch.utils.tensorboard import SummaryWriter
from utils.util import AverageMeter, Metric, accuracy, adjust_learning_rate, format_time, set_seed
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

def load_weights(ckpt_path, model, optimizer, resume=True) -> int:
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    if resume:
        # load model & optimizer
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        raise ValueError

    start_epoch = checkpoint['epoch'] + 1
    print("Loaded. (epoch {})".format(checkpoint['epoch']))
    return start_epoch

def train(model1, model2, optimizer1, optimizer2, dataloader, criterion, epoch, scaler1, scaler2, cfg, logger=None, writer=None):
    model1.train() # 開啟batch normalization 和 dropout
    model2.train() # 開啟batch normalization 和 dropout

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()

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
        if scaler1 and scaler2:
            with autocast():
                logits1= model1(imgs)
                logits2= model2(imgs)
                loss1, loss2 = criterion(logits1, logits2, labels, epoch)
        else:
            logits1= model1(imgs)
            logits2= model2(imgs)
            loss1, loss2 = criterion(logits1, logits2, labels, epoch)

        losses1.update(loss1.item(), batch_size)
        losses2.update(loss2.item(), batch_size)

        # accurate
        acc1_1, acc5_1 = accuracy(logits1, labels, topk=(1,5))
        acc1_2, acc5_2 = accuracy(logits2, labels, topk=(1,5))

        top1_1.update(acc1_1.item(), batch_size)
        top1_2.update(acc1_2.item(), batch_size)

        # compute gradient and do SGD step
        optimizer1.zero_grad()
        if scaler1:
            scaler1.scale(loss1).backward()
            scaler1.step(optimizer1)
            scaler1.update()
        else:
            loss1.backward()
            optimizer1.step()

        optimizer2.zero_grad()    
        if scaler2:
            scaler2.scale(loss2).backward()
            scaler2.step(optimizer2)
            scaler2.update()
        else:
            loss2.backward()
            optimizer2.step()
        # measure elapsed time
        batch_time.update(time.time() - iter_end)
        iter_end = time.time()

        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:  # cfg.rank == 0:
            lr1 = optimizer1.param_groups[0]['lr']
            lr2 = optimizer2.param_groups[0]['lr']
            logger.info(f'Epoch[epoch][idx/iter] [{epoch}][{idx+1:>4d}/{num_iter}] - '
                        f'batch_time: {batch_time.avg:.3f}, '
                        f'lr1: {lr1:.5f}, '
                        f'loss1(loss avg): {loss1:.3f}({losses1.avg:.3f}), '
                        f'train_Acc1@1: {top1_1.avg:.3f}, '
                        f'lr2: {lr1:.5f}, '
                        f'loss2(loss avg): {loss2:.3f}({losses2.avg:.3f}), '
                        f'train_Acc2@1: {top1_2.avg:.3f}'
            )
    if logger is not None: 
        now = time.time()
        epoch_time = format_time(now - epoch_end)
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss1: {losses1.avg:.3f}, '
                    f'train_Acc1@1: {top1_1.avg:.3f}, '
                    f'train_loss2: {losses2.avg:.3f}, '
                    f'train_Acc2@1: {top1_2.avg:.3f}'
        )
    
    if writer is not None:
        lr1 = optimizer1.param_groups[0]['lr']
        lr2 = optimizer1.param_groups[0]['lr']
        writer.add_scalar('Train/lr1', lr1, epoch)
        writer.add_scalar('Train/loss1', losses1.avg, epoch)
        writer.add_scalar('Train/acc1@1', top1_1.avg, epoch)
        writer.add_scalar('Train/lr2', lr2, epoch)
        writer.add_scalar('Train/loss2', losses2.avg, epoch)
        writer.add_scalar('Train/acc2@1', top1_2.avg, epoch)

def valid(model1, model2, dataloader, criterion, epoch, cfg, logger, writer):
    model1.eval() # 開啟batch normalization 和 dropout
    model2.eval() # 開啟batch normalization 和 dropout

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()

    pred1 = []
    pred2 = []
    target = []
    metrix1 = Metric(cfg.num_classes)
    metrix2 = Metric(cfg.num_classes)
    
    end = time.time()

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            batch_size = targets.shape[0]

            # forward
            logits1 = model1(images)
            logits2 = model2(images)
            loss1, loss2 = criterion(logits1, logits2, targets, epoch)

            acc1_1, acc5_1 = accuracy(logits1, targets, topk=(1,5))
            acc1_2, acc5_2 = accuracy(logits2, targets, topk=(1,5))

            # update metric
            losses1.update(loss1.item(), batch_size)
            top1_1.update(acc1_1.item(), batch_size)
            top5_1.update(acc5_1.item(), batch_size)

            losses2.update(loss2.item(), batch_size)
            top1_2.update(acc1_2.item(), batch_size)
            top5_2.update(acc5_2.item(), batch_size)

            pred1.append(logits1.detach().copy())
            pred2.append(logits2.detach().copy())
            target.append(targets)

    
    epoch_time = format_time(time.time() - end)

    if logger is not None:
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'valid_loss1: {losses1.avg:.3f}, '
                    f'valid_Acc1@1: {top1_1.avg:.3f}, '
                    f'valid_Acc1@5: {top5_1.avg:.3f}, '
                    f'valid_loss2: {losses2.avg:.3f}, '
                    f'valid_Acc2@1: {top1_2.avg:.3f}, '
                    f'valid_Acc2@5: {top5_2.avg:.3f}'
        )

    pred1 = torch.cat(pred1)
    pred2 = torch.cat(pred2)
    target = torch.cat(target)

    metrix1.update(pred1, target)
    metrix2.update(pred2, target)
    f1_score1 = metrix1.f1_score('none')
    f1_score2 = metrix2.f1_score('none')
    print(f'======== f1_score1 ========')
    print(f1_score1)
    print(f'======== f1_score2 ========')
    print(f1_score2)

    if writer is not None:
        writer.add_scalar('Valid/loss1', losses1.avg, epoch)
        writer.add_scalar('Valid/acc1@1', top1_1.avg, epoch)
        writer.add_scalar('Valid/loss2', losses2.avg, epoch)
        writer.add_scalar('Valid/acc2@1', top1_2.avg, epoch)

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
    ##### model 1 #####
    # build model
    model1 = build_model(cfg.model.copy())
    model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1)
    model1.cuda()
    model1 = torch.nn.parallel.DistributedDataParallel(model1, device_ids=[cfg.local_rank], find_unused_parameters=False)
    # build optimizer
    optimizer1 = build_optimizer(cfg.optimizer, model1.parameters())
    ##### model 2 #####
    # build model
    model2 = build_model(cfg.model.copy())
    model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2)
    model2.cuda()
    model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[cfg.local_rank], find_unused_parameters=False)
    # build optimizer
    optimizer2 = build_optimizer(cfg.optimizer, model2.parameters())

    # build criterion
    criterion = build_loss(cfg.loss).cuda()
    
    start_epoch = 1

    cudnn.benchmark = True

    if cfg.amp:
        scaler1 = GradScaler()
        scaler2 = GradScaler()
    else:
        scaler1 = None
        scaler2 = None
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer1, epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer2, epoch)

        # train; all processes
        train(model1, model2, optimizer1, optimizer2, train_loader, criterion, epoch, scaler1, scaler2, cfg, logger, writer)
        
        valid(model1, model2, valid_loader, criterion, epoch, cfg, logger, writer)

        # save ckpt; master process
        if rank == 0 and epoch % cfg.save_interval == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state1': optimizer1.state_dict(),
                'model_state1': model1.state_dict(),
                'optimizer_state2': optimizer2.state_dict(),
                'model_state2': model2.state_dict(),
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