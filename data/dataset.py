import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

def download_dataset(data_dir: str, num_parts: int = 30) -> None:
    """Download the waveform inversion dataset from KaggleHub."""
    try:
        import google.colab
        os.environ["DISABLE_COLAB_CACHE"] = "true"
        print("Running in Colab: DISABLE_COLAB_CACHE set.")
    except ImportError:
        print("Not in Colab: skipping DISABLE_COLAB_CACHE.")

    if os.path.exists(data_dir):
        for _, _, files in os.walk(data_dir):
            if any(f.endswith(".npy") for f in files):
                return

    os.makedirs(data_dir, exist_ok=True)
    import kagglehub
    os.environ["KAGGLEHUB_CACHE"] = data_dir

    for i in range(1, num_parts + 1):
        path = kagglehub.dataset_download(
            f"seshurajup/waveform-inversion-{i}"
        )
    print("Downloaded dataset to:", path)


class SeismicMelSpectrogramDataset(Dataset):
    def __init__(self, data_files, label_files=None, is_test=False):
        self.data_files = data_files
        self.label_files = label_files
        self.is_test = is_test
        self.mel_transform = MelSpectrogram(
            sample_rate=1000, n_fft=256, win_length=256, hop_length=14, n_mels=72
        )
        self.db_transform = AmplitudeToDB()

    def compute_mel(self, data_sample):
        spec_list = []
        for src in range(data_sample.shape[0]):
            receiver_mels = []
            for rcv in range(data_sample.shape[2]):
                sig = torch.tensor(data_sample[src, :, rcv], dtype=torch.float32)
                mel = self.mel_transform(sig)
                mel_db = self.db_transform(mel).numpy()
                mel_db = np.clip(
                    mel_db, np.percentile(mel_db, 1), np.percentile(mel_db, 99)
                )
                mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
                mel_db = mel_db.T
                receiver_mels.append(mel_db)
            spec_concat = np.concatenate(receiver_mels, axis=1)
            spec_list.append(spec_concat)
        return np.stack(spec_list, axis=0)

    def __getitem__(self, idx):
        file_idx = idx // 500
        sample_idx = idx % 500

        data = np.load(self.data_files[file_idx])
        data_sample = data[sample_idx]

        spec_array = self.compute_mel(data_sample)
        spec_tensor = torch.tensor(spec_array, dtype=torch.float32)

        if self.is_test:
            file_id = os.path.basename(self.data_files[file_idx]).replace(".npy", "")
            return spec_tensor, file_id

        label_sample = np.load(self.label_files[file_idx])[sample_idx]
        label_tensor = torch.tensor(label_sample, dtype=torch.float32)
        return spec_tensor, label_tensor

    def __len__(self):
        return len(self.data_files) * 500


def build_dataset(
    data_dir,
    batch_size,
    val_split=0.8,
    num_workers=4,
    pin_memory=True,
    num_parts=30,
    families=None,
):
    download_dataset(data_dir, num_parts)

    default_families = [
        "FlatVel_A",
        "FlatVel_B",
        "Style_A",
        "Style_B",
        "CurveVel_A",
        "CurveVel_B",
        "FlatFault_A",
        "FlatFault_B",
        "CurveFault_A",
        "CurveFault_B",
    ]
    families = families or default_families

    vel_families = {
        "FlatVel_A",
        "FlatVel_B",
        "Style_A",
        "Style_B",
        "CurveVel_A",
        "CurveVel_B",
    }

    data_files, label_files = [], []
    for version in range(1, num_parts + 1):
        cache_dir = os.path.join(
            data_dir, f"datasets/seshurajup/waveform-inversion-{version}", "versions", "1"
        )
        for fam in families:
            fam_dir = os.path.join(cache_dir, fam)
            print("Checking paths like:", fam_dir)

            if fam in vel_families:
                data_files += sorted(glob(os.path.join(fam_dir, "data", "*.npy")))
                label_files += sorted(glob(os.path.join(fam_dir, "model", "*.npy")))
            else:
                data_files += sorted(glob(os.path.join(fam_dir, "seis*.npy")))
                label_files += sorted(glob(os.path.join(fam_dir, "vel*.npy")))

    print(f"✅ Found {len(data_files)} data files and {len(label_files)} label files")

    split = int(val_split * len(data_files))
    train_ds = SeismicMelSpectrogramDataset(data_files[:split], label_files[:split])
    val_ds = SeismicMelSpectrogramDataset(data_files[split:], label_files[split:])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader

