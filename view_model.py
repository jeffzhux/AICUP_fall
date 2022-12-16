import torch
from utils.util import Metric
from sklearn import metrics
import numpy as np

models_pred = torch.load(f'./test_experiment/V2S_ensemble/pred_class_80.pt')
target = torch.load(f'./test_experiment/V2S_ensemble/target.pt')

def get_wp(models):
    pred_class = None
    for name in models:
        if pred_class is None:
            pred_class = models_pred[name]
        else:
            pred_class += models_pred[name]
    pred_class = pred_class / len(models)

    metrix = Metric(pred_class.size(-1))
    metrix.update(pred_class, target)
    f1 = metrix.f1_score('none')
    wp = metrix.accuracy('mean')
    return wp.item(), f1[-1].item()



# # keys = ['20221210_203426', '20221212_144039', '20221206_092628'] max_value = 0.8879038095474243, 0.893463671207428, 0.8956134915351868

# keys = ['20221210_203426_ema', '20221212_144039', '20221209_091753_ema']
# # max_value = 0.6798029541969299, 0.6777251362800598
# ck = []
# max_value = 0
# for key, model in models_pred.items():
#     a = {k :models_pred [k]for k in ck}
#     if key not in keys:
#         a.update({key: model})
#         tck = ck.extend(key)
#         max_value = max(get_wp(a), max_value)
#         print(get_wp(a), key)
models = np.array(list(models_pred.keys()))
max_models = None
maxInclass = 0
maxOther = 0
for i in [1,2,4,8,16,32,64,128]:#range(1,256):
    mask = np.array(list(str(format(i, 'b')).zfill(8)))

    models_list = models[np.where(mask == "1")[0]]
    in_score, other_score = get_wp(models_list)
    print(models_list, in_score)
    if maxOther < other_score:
        maxInclass = in_score
        maxOther = other_score
        max_models = models_list
    
print(max_models)
print(maxInclass)
print(maxOther)