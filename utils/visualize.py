import matplotlib.pyplot as plt
import torch
import wandb

def visualize_batch(x, y, out, epoch, max_visualize=4):
    for b in range(min(x.size(0), max_visualize)):
        vmin = min(y[b].min().item(), out[b].min().item())
        vmax = max(y[b].max().item(), out[b].max().item())
        abs_err = torch.abs(out[b] - y[b])

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(x[b].mean(0).cpu().numpy(), aspect='auto', cmap='plasma')
        plt.title("Input Mel (avg)")
        plt.colorbar()

        plt.subplot(1, 4, 2)
        plt.imshow(y[b].cpu().squeeze(), vmin=vmin, vmax=vmax, cmap='viridis')
        plt.title("GT Velocity")
        plt.colorbar()

        plt.subplot(1, 4, 3)
        plt.imshow(out[b].cpu().squeeze(), vmin=vmin, vmax=vmax, cmap='viridis')
        plt.title("Prediction")
        plt.colorbar()

        plt.subplot(1, 4, 4)
        plt.imshow(abs_err.cpu().squeeze(), cmap='hot')
        plt.title(f"Abs Error (MAE={abs_err.mean():.2f})")
        plt.colorbar()

        plt.suptitle(f"Epoch {epoch+1} | Sample {b+1}", fontsize=14)
        plt.tight_layout()
        wandb.log({f"val_image_{epoch+1}_{b+1}": wandb.Image(plt.gcf())})
        plt.show()