import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import yaml
import torch
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from huggingface_hub import login, create_repo, upload_folder
from dotenv import load_dotenv

from data.dataset import build_dataset
from models import build_model
from utils.optim import init_opt
from utils.visualize import visualize_batch

VEL_MIN, VEL_MAX = 1500.0, 4500.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--repo_id', type=str, help="HuggingFace repo ID")
    parser.add_argument('--num_parts', type=int, help='Number of dataset parts to download')
    parser.add_argument(
        '--families', nargs='*', type=str, help='Dataset families to use'
    )
    args = parser.parse_args()

    # Load and override args from YAML
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if getattr(args, key, None) is None:
                    setattr(args, key, value)
    return args


def train_one_epoch(model, ema_model, loader, optimizer, scaler, scheduler, device):
    model.train()
    losses = []
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        with autocast(device_type='cuda',dtype=torch.bfloat16):
            out = model(x)
            loss = F.l1_loss(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        if ema_model:
            ema_model.update(model)

        losses.append(loss.item())
    return np.mean(losses)


def validate(model, loader, epoch, device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda',dtype=torch.bfloat16):
                out = model(x)
                loss = F.l1_loss(out, y)
            val_losses.append(loss.item())
            if epoch % 10 == 0 and i == 0:
                visualize_batch(x, y, out, epoch)
    return np.mean(val_losses)


def main(args):
    load_dotenv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader = build_dataset(
        args.data_dir,
        args.batch_size,
        num_parts=args.num_parts,
        families=args.families,
    )

    model, ema_model = build_model(device)

    optimizer, scaler, scheduler = init_opt(
        model, args.lr, args.epochs, len(train_loader)
    )

    wandb.init(project="openfwi-ensemble", config=vars(args))

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n🌟 Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, ema_model, train_loader, optimizer, scaler, scheduler, device)
        val_loss = validate(ema_model.module, val_loader, epoch, device)

        wandb.log({"epoch": epoch+1, "train_mae": train_loss, "val_mae": val_loss, "lr": scheduler.get_last_lr()[0]})

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(ema_model.module.state_dict(), "best_model_ema.pth")
            print(f"💾 Saved new best model (Val MAE: {val_loss:.2f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./openfwi_best_model_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.rename("best_model_ema.pth", os.path.join(save_dir, "pytorch_model.bin"))

    config = {
        "model_name": "EnsembleNet",
        "input_channels": 350,
        "output": "velocity",
        "velocity_range": [VEL_MIN, VEL_MAX],
        "image_size": [70, 70],
        "saved_at": timestamp
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(f"# OpenFWI Best Model\nVal MAE: {best_loss:.2f}\n")

    repo_id = args.repo_id or f"openfwi_ensemble_{timestamp}"
    create_repo(repo_id, private=True, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=save_dir, commit_message="upload best model")
    print(f"✅ Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

