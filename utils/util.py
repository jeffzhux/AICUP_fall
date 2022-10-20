import math
from matplotlib.pyplot import table
import numpy as np
import torch
import torch.nn.functional as F
import random

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = []
        self.t = []
        self.tp = None
        self.fp = None
        self.fn = None
        self.tn = None
    def update(self, y, t):
        '''Update with batch outputs and labels.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        '''
        maxk = max((1,))
        _, y = y.topk(maxk, 1, True, True)
        y = torch.squeeze(y)
        
        self.y.append(y)
        self.t.append(t)
        self.tp = None
        self.fp = None
        self.fn = None
        self.tn = None
    def _process(self, y, t):
        '''Compute TP, FP, FN, TN.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''

        tp = torch.empty(self.num_classes)
        fp = torch.empty(self.num_classes)
        fn = torch.empty(self.num_classes)
        tn = torch.empty(self.num_classes)

        for i in range(self.num_classes):
            tp[i] = ((y == i) & (t == i)).sum().item()
            fp[i] = ((y == i) & (t != i)).sum().item()
            fn[i] = ((y != i) & (t == i)).sum().item()
            tn[i] = ((y != i) & (t != i)).sum().item()
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        return tp, fp, fn, tn

    def accuracy(self, reduction='mean'):
        '''Accuracy = (TP+TN) / (P+N).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) accuracy.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)

        if self.tp is not None:
            tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        else:
            tp, fp, fn, tn = self._process(y, t)

        if reduction == 'none':
            acc = tp / (tp + fn)
        else:
            acc = tp.sum() / (tp + fn).sum()
        return acc

    def recall(self, reduction='mean'):
        '''Recall = TP / (TP + FN)
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) recall.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)

        if self.tp is not None:
            tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        else:
            tp, fp, fn, tn = self._process(y, t)

        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0
        if reduction == 'mean':
            recall = recall.mean()
        return recall
    def weighted_precision(self, threshold=0.7):
        '''
        F1-score of each class must be greater than threshold
        (sum of precision of class which F1-score is greater than threshold * (TP + FN)) / Total Image Count

        '''
        if not self.y or not self.t:
            return
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        num_of_image = t.size(0)

        if self.tp is not None:
            tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        else:
            tp, fp, fn, tn = self._process(y, t)

        f1_score = self.f1_score('none')
        precision = self.precision('none')
        wp = torch.masked_select((precision * (tp+fn)), f1_score.ge(threshold)).sum() / num_of_image
        return wp

    def precision(self, reduction='mean'):
        '''Precision = TP / (TP+FP).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) precision.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        
        if self.tp is not None:
            tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        else:
            tp, fp, fn, tn = self._process(y, t)

        prec = tp / (tp + fp)
        prec[torch.isnan(prec)] = 0
        if reduction == 'mean':
            prec = prec.mean()
        return prec
    def f1_score(self, reduction='mean'):
        '''f1-score = 2 x precision x recall / (precision + recall).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) precision.
        '''
        recall = self.recall(reduction)
        precision = self.precision(reduction)
        return 2 * precision * recall / (precision+recall)

    def confusion_matrix(self):
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[j][i] = ((y == i) & (t == j)).sum().item()
        return matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrackMeter(object):
    """Compute and store values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.sum = 0
        self.avg = 0
        self.max_val = float('-inf')
        self.max_idx = -1

    def update(self, val, idx=None):
        self.data.append(val)
        self.sum += val
        self.avg = self.sum / len(self.data)
        if val > self.max_val:
            self.max_val = val
            self.max_idx = idx if idx else len(self.data)

    def last(self, k):
        assert 0 < k <= len(self.data)
        return sum(self.data[-k:]) / k


def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        res = []
        if len(target.size()) == 1: # general 
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        else: # one hot encoding
            target = target > 1e-3
            for k in topk:
                correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
                res.append(correct.sum().mul_(100.0 / target.sum()))
        return res

def other_accuracy(pred, target, topk=(1,)):

    pred_softmax = F.softmax(pred, dim=1) # do softmax
    pred_score = (-pred_softmax * torch.log(pred_softmax)).sum(-1)
    pred_score = torch.special.entr(pred_softmax).sum(-1) # do entropy
    others = torch.where(pred_score > 2.3, 1, 0).view(-1, 1)
    pred_softmax = torch.cat((pred_softmax, others), dim=-1)

    print(target.size())
    target_score = (-target * torch.log(target)).sum(-1) # do entropy
    target_others = torch.where(target_score > 2.3, 1, 0).view(-1, 1)
    target = torch.cat((target, target_others), dim=-1)

    return accuracy(pred_softmax, target, topk)

    

def _get_lr(cfg, step):
    lr = cfg.lr
    if cfg.type == 'Cosine':  # Cosine Anneal
        start_step = cfg.get('start_step', 1)
        eta_min = max(lr * cfg["decay_rate"], cfg.get('min', 0))
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * (step - start_step) / cfg.steps)) / 2
    elif cfg.type == 'MultiStep':  # MultiStep
        num_steps = np.sum(step > np.asarray(cfg.decay_steps))
        lr = lr * (cfg.decay_rate ** num_steps)
    else:
        raise NotImplementedError(cfg.type)
    return lr


def adjust_learning_rate(cfg, optimizer, step, batch_idx=0, num_batches=100):
    start_step = cfg.get('start_step', 1)
    if step < cfg.get('warmup_steps', 0) + start_step:
        warmup_to = _get_lr(cfg, cfg.warmup_steps + 1)
        p = (step - start_step + batch_idx / num_batches) / cfg.warmup_steps
        lr = cfg.warmup_from + p * (warmup_to - cfg.warmup_from)
    else:
        lr = _get_lr(cfg, step)

    # update optimizer lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr_simsiam(cfg, optimizer, step):
    init_lr = cfg.lr
    lr = _get_lr(cfg, step)
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = lr


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    # millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    # if millis > 0 and i <= 2:
    #     f += str(millis) + 'ms'
    #     i += 1
    if f == '':
        f = '0ms'
    return f


def get_activation(name, activation):
    '''
        >>> activation = {}
        >>> for name, layer in model.named_modules():
            layer.register_forward_hook(get_activation(name))
    '''
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook