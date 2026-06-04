import json
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Sampler
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
        T.Spacingd(keys=image_keys + label_keys, pixdim=(1.0, 1.0, 1.0),
                   mode=["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)),
        # Ensure uniform spatial size — adjust to your data
        T.ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_shape),
        # Intensity normalise images only
        T.NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        T.ToTensord(keys=all_keys),
    ]

    if split == "train":
        augment = [
            # ── Spatial (applied identically to both timepoints and labels) ───
            #
            # Mirror transform: flip each axis independently with p=0.5
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
            # Rotation ±30° (0.52 rad) on all axes, p=0.2
            # Images: bilinear interpolation; labels: nearest-neighbour
            T.RandRotated(
                keys=all_keys,
                range_x=0.52, range_y=0.52, range_z=0.52,
                prob=0.2,
                mode=["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys),
                padding_mode="zeros",
            ),
            # Scaling 0.7–1.4, p=0.2; keep_size crops/pads back to original shape
            T.RandZoomd(
                keys=all_keys,
                min_zoom=0.7, max_zoom=1.4,
                prob=0.2,
                mode=["trilinear"] * len(image_keys) + ["nearest"] * len(label_keys),
                keep_size=True,
            ),

            # ── Intensity (images only — never applied to labels) ─────────────
            #
            # Gaussian noise: nnUNet uses variance ~ U(0, 0.1) → std up to 0.316
            T.RandGaussianNoised(keys=image_keys, mean=0.0, std=0.1, prob=0.1),
            # Gaussian blur: sigma U(0.5, 1.0), p=0.2
            T.RandGaussianSmoothd(
                keys=image_keys,
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
                prob=0.2,
            ),
            # Multiplicative brightness: multiplier U(0.75, 1.25), p=0.15
            # RandScaleIntensity multiplies by (1 + factor), so factors=0.25 → [0.75, 1.25]
            T.RandScaleIntensityd(keys=image_keys, factors=0.25, prob=0.15),
            # Contrast adjustment: gamma U(0.75, 1.25), p=0.15
            T.RandAdjustContrastd(keys=image_keys, gamma=(0.75, 1.25), prob=0.15),
            # Low-resolution simulation: downsample then upsample, p=0.25
            T.RandSimulateLowResolutiond(
                keys=image_keys,
                zoom_range=(0.5, 1.0),
                prob=0.25,
            ),
            # Gamma correction: gamma U(0.7, 1.5), p=0.3
            T.RandAdjustContrastd(keys=image_keys, gamma=(0.7, 1.5), prob=0.3),
        ]
        return T.Compose(base + augment)

    return T.Compose(base)


# ------------------------------------------------------------------ #
# 3. Foreground oversampling sampler
# ------------------------------------------------------------------ #

class ForegroundOversampledSampler(Sampler):
    """
    Produces len(dataset) indices per epoch, where a fixed fraction
    (`oversample_rate`, default 0.33) are drawn from samples that contain
    at least one foreground lesion voxel in the target label (label2).
    The remaining indices are drawn uniformly at random from the full dataset.

    This mirrors nnUNet's foreground oversampling strategy and helps the model
    see lesion-positive samples more frequently — critical for small, sparse
    MS lesions where many volumes may be lesion-free.

    The foreground/background split is computed once at construction by
    scanning all label2 files; subsequent epochs reuse this index.
    """

    def __init__(self, dataset: LongitudinalLesionDataset,
                 label_key: str = "label2",
                 oversample_rate: float = 0.33):
        self.n           = len(dataset)
        self.n_fg        = round(self.n * oversample_rate)
        self.n_rnd       = self.n - self.n_fg

        # Scan label files once to identify foreground samples
        print(f"ForegroundOversampledSampler: scanning {self.n} label files …")
        fg_indices, bg_indices = [], []
        for i, entry in enumerate(dataset.samples):
            vol = nib.load(entry[label_key]).get_fdata(dtype=np.float32)
            (fg_indices if vol.max() > 0 else bg_indices).append(i)

        if not fg_indices:
            raise RuntimeError(
                "ForegroundOversampledSampler: no foreground samples found. "
                "Check that label2 files contain lesion voxels."
            )

        self.fg_indices  = np.array(fg_indices)
        self.all_indices = np.arange(self.n)
        print(f"  → {len(fg_indices)} foreground / {len(bg_indices)} background samples.")

    def __iter__(self):
        # Draw foreground-guaranteed indices (with replacement to handle small fg sets)
        fg_draw  = np.random.choice(self.fg_indices, size=self.n_fg,  replace=True)
        # Draw the rest uniformly from the full dataset (without replacement)
        rnd_draw = np.random.choice(self.all_indices, size=self.n_rnd, replace=False)
        indices  = np.concatenate([fg_draw, rnd_draw])
        np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.n


# ------------------------------------------------------------------ #
# 4. Custom collate — handles the string metadata fields
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
# 5. DataLoader factory
# ------------------------------------------------------------------ #

def get_dataloaders(json_path: str,
                    target_shape=(192, 192, 192),
                    batch_size: int = 2,
                    num_workers: int = 4,
                    oversample_rate: float = 0.33):
    """
    Returns (train, val, test) DataLoaders.

    The training loader uses ForegroundOversampledSampler so that
    `oversample_rate` fraction of each epoch's samples are guaranteed to
    contain at least one foreground lesion voxel (label2 > 0).
    Val and test loaders iterate sequentially without oversampling.
    """
    loaders = {}
    for split in ("train", "validation", "test"):
        ds = LongitudinalLesionDataset(
            json_path  = json_path,
            split      = split,
            transform  = get_transforms(split, target_shape),
        )
        if split == "train":
            sampler = ForegroundOversampledSampler(
                ds, label_key="label2", oversample_rate=oversample_rate
            )
            loaders[split] = DataLoader(
                ds,
                batch_size  = batch_size,
                sampler     = sampler,        # replaces shuffle=True
                num_workers = num_workers,
                pin_memory  = True,
                collate_fn  = longitudinal_collate,
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = True,
                collate_fn  = longitudinal_collate,
            )

    return loaders["train"], loaders["validation"], loaders["test"]