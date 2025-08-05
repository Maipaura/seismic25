import torch
from torch.cuda.amp import GradScaler

from .scheduler import ConstantCosineLR


def init_opt(model, lr, epochs, steps_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scaler = GradScaler()
    scheduler = ConstantCosineLR(optimizer, total_steps=epochs * steps_per_epoch)
    return optimizer, scaler, scheduler

