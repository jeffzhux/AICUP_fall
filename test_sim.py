import os
import platform
import argparse
import time
import csv
import numpy as np

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
from utils.util import visualize
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
def iterate_data(model, model_ema, dataloader, cfg):
    
    pred_class = []
    target = []
    features = []
    for idx, (images, labels, loc, text) in enumerate(dataloader):
        images = images.float().cuda()
        labels = labels.cuda()
        loc = loc.cuda()

        # forward
        logits = model(images, loc, text)
        feature = model.module.image_encoder.features(images)
        feature = model.module.image_encoder.avgpool(feature)
        # logits = model_ema(images, loc, text)
        pred_class.append(logits.cpu())
        target.append(labels.cpu())
        features.append(feature)

    pred_class = torch.cat(pred_class)
    target = torch.cat(target)
    features = torch.cat(features)
    return pred_class, target, features

@torch.no_grad()
def run_eval(model, model_ema, test_loader, dataset, cfg):
    
    # pred_class, target, img_features = iterate_data(model, model_ema, test_loader, cfg)
    # torch.save(pred_class,'./pred_class.pt')
    # torch.save(target,'./target.pt')
    # torch.save(img_features,'./features.pt')
    
    idx_to_classes = cfg.data.idx_to_classes
    img_features = None
    pred_class = torch.load('./pred_class.pt')
    target = torch.load('./target.pt')
    img_features = torch.load('./features.pt')
    soft_class = F.softmax(pred_class, dim=-1).cpu()
    open_set_pred = torch.sum(-soft_class * torch.log(soft_class + 1e-5), dim=-1)
    # max_index, max_value = torch.max(open_set_pred, dim=-1)
    
    # pred_class[open_set_pred.gt(2.5),32] = 10
    
    
    metrix = Metric(pred_class.size(-1))
    metrix.update(pred_class, target)
    confuse_matrix = metrix.confusion_matrix().numpy()
    acc = metrix.accuracy('mean')
    wp = metrix.weighted_precision(0.7)
    if cfg.draw:

        max_val, max_idx = torch.max(soft_class, dim=-1)
        file_name_list = np.array(list(map(lambda x: x[0].split('\\')[-1], dataset.imgs)))
        pred = np.array(list(map(lambda x: idx_to_classes[x], max_idx.tolist())))
        lab = np.array(list(map(lambda x: idx_to_classes[x], target.tolist())))
        max_val = max_val.cpu().numpy()
        others_error_idx = np.setxor1d(np.where(lab=='others'), np.where(pred=='others'))

        assert len(pred_class)== len(file_name_list), f'pred len is {len(pred_class)}, but file_name_list len is {len(file_name_list)}'
        with open(os.path.join(cfg.work_dir, f'others_error.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([['filename', 'label', 'prd_label(Top1)']])
            for fn, l,pl,s in zip(file_name_list[others_error_idx], lab[others_error_idx], pred[others_error_idx], max_val[others_error_idx]):
                writer.writerows([[fn, l, f'{pl}({s:.3f})']])
        exit()
        fig, ax = plt.subplots(figsize=(20,20))
        ax.ticklabel_format(style='plain', useOffset=False)
        cm_display = metrics.ConfusionMatrixDisplay(confuse_matrix, display_labels=dataset.class_to_idx.keys())
        cm_display.plot(ax = ax, values_format='.20g', cmap='Blues')
        plt.xticks(rotation=90)
        cm_display.figure_.savefig(os.path.join(cfg.work_dir, f'confuse_metrix.png'))
        tSNE_filename = os.path.join(cfg.work_dir, f'TSNE.png')
        visualize(img_features.cpu(), target, dataset.class_to_idx, tSNE_filename)

    print(metrics.classification_report(target.cpu().tolist(), torch.argmax(pred_class, dim=-1).cpu().tolist(), target_names=dataset.class_to_idx.keys()))
    print(f'acc : {acc}')
    print(f'wp : {wp.item()}')

@torch.no_grad()
def run_inference(model, model_ema, dataloader, dataset, cfg):
    pred_class = []
    idx_to_classes = cfg.data.idx_to_classes

    for idx, (images, _, loc, text) in enumerate(dataloader):
        images = images.float().cuda()
        loc = loc.cuda()
        text = text.cuda()
        # forward
        logits = model(images,loc, text)
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
    model_without_ddp = model.module
    

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    model_ema = build_ema_model(model_without_ddp, cfg)
    if cfg.load:
        load_weights(cfg.load, model_without_ddp, model_ema)
    model_ema = torch.nn.parallel.DistributedDataParallel(model_ema, device_ids=[cfg.local_rank])
    print(f"==> Start testing ....")
    model.eval()
    # with torch.inference_mode:
    if cfg.output_file_name is not None:
        run_inference(model, model_ema, test_loader, test_set, cfg)
    else:
        run_eval(model, model_ema, test_loader, test_set, cfg)
        
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