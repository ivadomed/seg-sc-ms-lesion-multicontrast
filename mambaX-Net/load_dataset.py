import json
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from monai import transforms as T


# ------------------------------------------------------------------ #
# 1. Dataset class
# ------------------------------------------------------------------ #

class LongitudinalLesionDataset(Dataset):
    """
    Loads consecutive image pairs for longitudinal MS lesion segmentation.
    Each sample is a dict with keys: image1, label1, image2, label2.
    All volumes are returned as float32 tensors with a channel dim: (1, H, W, D).
    """

    def __init__(self, json_path: str, split: str = "train", transform=None):
        """
        Args:
            json_path  : path to the dataset JSON produced by create_datalist.py
            split      : "train", "validation", or "test"
            transform  : optional MONAI/callable transform applied to each sample dict
        """
        with open(json_path) as f:
            dataset = json.load(f)

        assert split in ("train", "validation", "test"), \
            f"split must be train/validation/test, got {split}"

        self.samples   = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _load_nifti(self, path: str) -> torch.Tensor:
        """Load a NIfTI file and return a (1, H, W, D) float32 tensor."""
        vol = nib.load(path).get_fdata(dtype=np.float32)
        return torch.from_numpy(vol).unsqueeze(0)   # add channel dim

    def __getitem__(self, idx: int) -> dict:
        entry = self.samples[idx]

        sample = {
            "image1":   self._load_nifti(entry["image1"]),
            "label1":   self._load_nifti(entry["label1"]),
            "image2":   self._load_nifti(entry["image2"]),
            "label2":   self._load_nifti(entry["label2"]),
            # metadata — not tensors, kept out of the collate stack
            "subject":  entry["subject"],
            "contrast": entry["contrast"],
            "session1": entry["session1"],
            "session2": entry["session2"],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# ------------------------------------------------------------------ #
# 2. MONAI transforms
# ------------------------------------------------------------------ #

def get_transforms(split: str, target_shape=(192, 192, 192)):
    """
    Returns a MONAI Compose for training or inference.
    All keys operate on image1/label1 and image2/label2 in parallel
    so spatial augmentations are applied IDENTICALLY to both timepoints.
    """
    image_keys = ["image1", "image2"]
    label_keys = ["label1", "label2"]
    all_keys   = image_keys + label_keys

    # --- transforms shared across splits ---
    base = [
        # Reorient to RPI (adjust to your data) — ensures consistent orientation across subjects
        T.Orientationd(keys=all_keys, axcodes="RPI", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        # Resample to 1mm iso
        T.Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        # Ensure uniform spatial size — adjust to your data
        T.ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_shape),
        # Intensity normalise images only
        T.NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        T.ToTensord(keys=all_keys),
    ]

    if split == "train":
        augment = [
            # Identical random flips applied to both timepoints
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
            # Intensity jitter on images only (not labels)
            T.RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
        ]
        return T.Compose(augment + base)

    return T.Compose(base)


# ------------------------------------------------------------------ #
# 3. Custom collate — handles the string metadata fields
# ------------------------------------------------------------------ #

def longitudinal_collate(batch: list) -> dict:
    """
    Default torch collate breaks on string fields.
    This stacks tensors and collects strings into lists.
    """
    tensor_keys = {"image1", "label1", "image2", "label2"}
    string_keys = {"subject", "contrast", "session1", "session2"}

    out = {}
    for key in tensor_keys:
        out[key] = torch.stack([b[key] for b in batch])
    for key in string_keys:
        out[key] = [b[key] for b in batch]
    return out


# ------------------------------------------------------------------ #
# 4. DataLoader factory
# ------------------------------------------------------------------ #

def get_dataloaders(json_path: str,
                    target_shape=(192, 192, 192),
                    batch_size: int = 2,
                    num_workers: int = 4):
    """
    Returns train, val, test DataLoaders ready for a training loop.
    """
    loaders = {}
    for split in ("train", "validation", "test"):
        ds = LongitudinalLesionDataset(
            json_path  = json_path,
            split      = split,
            transform  = get_transforms(split, target_shape),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
            collate_fn  = longitudinal_collate,
        )
        # print(f"{split}: {len(ds)} pairs → {len(loaders[split])} batches")

    return loaders["train"], loaders["validation"], loaders["test"]


# ------------------------------------------------------------------ #
# 5. Usage
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        json_path    = "dataset_2026-05-13.json",
        target_shape = (128, 128, 64),   # tune to your GPU memory
        batch_size   = 2,
        num_workers  = 4,
    )

    # Sanity check
    batch = next(iter(train_loader))
    print("image1 shape:", batch["image1"].shape)   # (B, 1, 128, 128, 64)
    print("label2 shape:", batch["label2"].shape)   # (B, 1, 128, 128, 64)
    print("subjects    :", batch["subject"])
    print("contrasts   :", batch["contrast"])