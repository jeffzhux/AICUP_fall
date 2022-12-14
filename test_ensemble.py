import os
import platform
import argparse
import time
import csv
import numpy as np
import copy

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
def iterate_data(models, dataloader, cfg):
    
    pred_class = {}
    features = {}
    target = []
    for name in models.keys():
        pred_class[name] = []
        features[name] = []

    for idx, (images, labels, loc, text) in enumerate(dataloader):
        batch_size = images[0].size(0)
        
        loc = loc.repeat(len(images), 1).cuda()
        text = text.repeat(len(images), 1)

        images = torch.cat(images, dim=0)
        images = images.float().cuda()
        
        # forward
        for name, model in models.items():

            logits = model(images, loc, text)
            logits = torch.softmax(logits, dim=-1)
            if logits.size(0) != batch_size:
                logits = logits.view(-1, batch_size, logits.size(-1))
                logits = 0.5 * logits[0] + 0.5 * torch.mean(logits[1:], dim=0)
                
            pred_class[name].append(logits.cpu())

            if cfg.draw:
                feature = model.module.image_encoder.features(images)
                feature = model.module.image_encoder.avgpool(feature)            
                features[name].append(feature.cpu())

        target.append(labels.cpu())
    
    for name in models.keys():
        pred_class[name] = torch.cat(pred_class[name])
        if cfg.draw:
            features[name] = torch.cat(features[name])
    target = torch.cat(target)

    
    torch.save(pred_class,f'{cfg.work_dir}/pred_class.pt')
    torch.save(target,f'{cfg.work_dir}/target.pt')
    if cfg.draw:
        torch.save(features,f'{cfg.work_dir}/features.pt')
    return pred_class, target, features

@torch.no_grad()
def run_eval(models, test_loader, dataset, cfg):
    
    if cfg.save_pred:
        print(f'save model predict')
        models_pred, target, models_features = iterate_data(models, test_loader, cfg)
    else:
        print(f'load model predict')
        models_pred = torch.load(f'{cfg.work_dir}/pred_class.pt')
        target = torch.load(f'{cfg.work_dir}/target.pt')
        models_features = torch.load(f'{cfg.work_dir}/features.pt')
    idx_to_classes = cfg.data.idx_to_classes

    pred_class = None
    for name, logits in models_pred.items():
        if pred_class is None:
            pred_class = logits
        else:
            pred_class += logits
    pred_class = pred_class / len(models)
    
    metrix = Metric(pred_class.size(-1))
    metrix.update(pred_class, target)
    confuse_matrix = metrix.confusion_matrix().numpy()
    acc = metrix.accuracy('mean')
    wp = metrix.weighted_precision(0.7)
    if cfg.draw:

        max_val, max_idx = torch.max(pred_class, dim=-1)
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
def run_inference(models, dataloader, dataset, cfg):
    pred_logit = {}
    idx_to_classes = cfg.data.idx_to_classes

    for name, model in models.items():
        pred_logit[name] = []

    for idx, (images, _, loc, text) in enumerate(dataloader):

        batch_size = images[0].size(0)
        
        loc = loc.repeat(len(images), 1).cuda()
        text = text.repeat(len(images), 1).cuda()

        images = torch.cat(images, dim=0)
        images = images.float().cuda()

        # forward
        for name, model in models.items():
            logits = model(images, loc, text)
            logits = torch.softmax(logits, dim=-1)
            if logits.size(0) != batch_size:
                logits = logits.view(-1, batch_size, logits.size(-1))
                logits = 0.5 * logits[0] + 0.5 * torch.mean(logits[1:], dim=0)
            pred_logit[name].append(logits)
            
    for name in models.keys():
        pred_logit[name] = torch.cat(pred_logit[name])

    torch.save(pred_logit, f'{cfg.work_dir}/pred_logit.pt')

    pred_class = None
    for name, logits in pred_logit.items():
        if pred_class is None:
            # pred_class = torch.softmax(logits, dim=-1)
            pred_class = logits
        else:
            # pred_class += torch.softmax(logits, dim=-1)
            pred_class += logits

    pred_class = pred_class / len(pred_logit)
    pred_class = torch.argmax(pred_class, dim=-1).cpu().tolist()

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
    torch.cuda.empty_cache()
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
    model_ensemble = {}
    for idx, load_path in enumerate(cfg.load):
        name = load_path.split('/')[-2]
        model_ensemble[name] = build_model(cfg.models[idx])
        model_ensemble[name].cuda()
        
        model_ensemble[name] = torch.nn.parallel.DistributedDataParallel(model_ensemble[name], device_ids=[cfg.local_rank])
        model_without_ddp = model_ensemble[name].module
        
        model_ensemble[f'{name}_ema'] = build_ema_model(model_without_ddp, cfg)
        load_weights(load_path, model_without_ddp, model_ensemble[f'{name}_ema'])
        model_ensemble[f'{name}_ema'] = torch.nn.parallel.DistributedDataParallel(model_ensemble[f'{name}_ema'], device_ids=[cfg.local_rank])

        model_ensemble[name].eval()
        model_ensemble[f'{name}_ema'].eval()
        
        if name not in ['20221210_203426_ema', '20221212_144039_ema']:
            # print(f'delete {name}')
            del model_ensemble[name]
        if f'{name}_ema' not in ['20221210_203426_ema', '20221212_144039_ema']:
            # print(f'delete {name}_ema')
            del model_ensemble[f'{name}_ema']
    
    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    
    # with torch.inference_mode:
    if cfg.output_file_name is not None:
        run_inference(model_ensemble, test_loader, test_set, cfg)
    else:
        run_eval(model_ensemble, test_loader, test_set, cfg)
        
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