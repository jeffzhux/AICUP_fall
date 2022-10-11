import torch
import torch.nn as nn

from utils.config import ConfigDict
class TestTimeAugmentation(nn.Module):
    def __init__(self, cfg:ConfigDict):
        super(TestTimeAugmentation, self).__init__()

        self.step = cfg.num_of_trans + 1 # original + num of augmentation
        self.merge_mode = cfg.merge_mode
        self.sharpen = cfg.sharpen
    def forward(self, logits):
        output = []
        if self.step > 1:
            for i in range(0, logits.shape[0], self.step):
                tran_logit = torch.__dict__[self.merge_mode](logits[i + 1:i + self.step], dim=0)
                logit = logits[i] * self.sharpen + (1 - self.sharpen) * tran_logit
                output.append(logit)

            output = torch.stack(output)
        else :
            output = logits
        return output
        