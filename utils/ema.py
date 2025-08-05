import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99):
        super().__init__()
        self.module = deepcopy(model).eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                ema_v.data.mul_(self.decay).add_(model_v.data, alpha=1. - self.decay)