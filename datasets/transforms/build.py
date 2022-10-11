from datasets import transforms

def build_transform(cfg):
    args = cfg.copy()
    func_name = args.pop('type')
    if func_name:
        return transforms.__dict__[func_name](**args)
    else:
        return None