"""
Dataset and dataloader for MambaXNet-SC-Lesion.

Each sample is a consecutive image pair (M0, M12) with five volumes:
    image1       – M0 image
    image2       – M12 image
    sc_mask1     – spinal cord mask at M0
    sc_mask2     – spinal cord mask at M12
    lesion_mask1 – lesion mask at M0
    lesion_mask2 – lesion mask at M12  (prediction target)

Expected dataset JSON structure:
    {
      "train":      [ { "image1": "...", "image2": "...",
                        "sc_mask1": "...", "sc_mask2": "...",
                        "lesion_mask1": "...", "lesion_mask2": "..." }, ... ],
      "validation": [ ... ],
      "test":       [ ... ]
    }

Author: Pierre-Louis Benveniste
"""

import json

import nibabel as nib
import numpy as np
import torch
from monai import transforms as T
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SCLesionDataset(Dataset):
    """
    Loads consecutive image pairs for SC-guided longitudinal lesion segmentation.
    All volumes are returned as float32 tensors with a channel dim: (1, H, W, D).
    """

    IMAGE_KEYS  = ["image1",    "image2"]
    MASK_KEYS   = ["sc_mask1",  "sc_mask2", "lesion_mask1", "lesion_mask2"]
    STRING_KEYS = ["subject",   "contrast", "session1",     "session2"]

    def __init__(self, json_path: str, split: str = "train", transform=None):
        assert split in ("train", "validation", "test"), \
            f"split must be train/validation/test, got {split!r}"

        with open(json_path) as f:
            self.samples = json.load(f)[split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_nifti(path: str) -> torch.Tensor:
        """Load a NIfTI file and return a (1, H, W, D) float32 tensor."""
        vol = nib.load(path).get_fdata(dtype=np.float32)
        return torch.from_numpy(vol).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict:
        entry = self.samples[idx]

        sample = {key: self._load_nifti(entry[key])
                  for key in self.IMAGE_KEYS + self.MASK_KEYS}
        sample.update({key: entry[key] for key in self.STRING_KEYS})

        if self.transform:
            sample = self.transform(sample)

        return sample


# ──────────────────────────────────────────────────────────────────────────────
# MONAI transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str, target_shape=(192, 192, 192)):
    """
    MONAI Compose for training or inference.

    Spatial transforms are applied identically to all volumes so that
    image/mask alignment is preserved across time points.
    """
    image_keys = ["image1",    "image2"]
    mask_keys  = ["sc_mask1",  "sc_mask2", "lesion_mask1", "lesion_mask2"]
    all_keys   = image_keys + mask_keys

    base = [
        T.Orientationd(keys=all_keys, axcodes="RPI",
                       labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        T.Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                   mode=["bilinear"] * len(image_keys) + ["nearest"] * len(mask_keys)),
        T.ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_shape),
        T.NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        T.ToTensord(keys=all_keys),
    ]

    if split == "train":
        augment = [
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
            T.RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
        ]
        return T.Compose(base + augment)

    return T.Compose(base)


# ──────────────────────────────────────────────────────────────────────────────
# Custom collate
# ──────────────────────────────────────────────────────────────────────────────

def sc_lesion_collate(batch: list) -> dict:
    """Stack tensors; collect string metadata into lists."""
    tensor_keys = {"image1", "image2", "sc_mask1", "sc_mask2",
                   "lesion_mask1", "lesion_mask2"}
    string_keys = {"subject", "contrast", "session1", "session2"}

    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out.update({k: [b[k] for b in batch] for k in string_keys})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(json_path: str,
                    target_shape=(192, 192, 192),
                    batch_size: int = 1,
                    num_workers: int = 4):
    """Return (train, val, test) DataLoaders."""
    loaders = {}
    for split in ("train", "validation", "test"):
        ds = SCLesionDataset(
            json_path = json_path,
            split     = split,
            transform = get_transforms(split, target_shape),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
            collate_fn  = sc_lesion_collate,
        )
    return loaders["train"], loaders["validation"], loaders["test"]
