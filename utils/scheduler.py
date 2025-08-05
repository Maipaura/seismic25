import math
from torch.optim.lr_scheduler import _LRScheduler

class ConstantCosineLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, pct_cosine=0.5, last_epoch=-1):
        self.milestone = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(total_steps - self.milestone, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        return [base_lr * factor for base_lr in self.base_lrs]