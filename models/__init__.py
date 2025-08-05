from .ensemble import EnsembleNet
from utils.ema import ModelEMA
import torch


def build_model(device: torch.device, decay: float = 0.99):
    """Construct the neural network, its EMA counterpart, and move both to ``device``."""
    model = EnsembleNet().to(device)
    ema_model = ModelEMA(model, decay=decay)
    ema_model.module.to(device)
    return model, ema_model

