"""
MambaXNet-SC-Lesion

Extends MambaXNet to accept two spinal cord (SC) masks alongside the previous
lesion mask, giving the model explicit registration anchors at both time points.

Five inputs:
    image_M12      – (B, 1, *spatial)  follow-up image           (current)
    image_M0       – (B, 1, *spatial)  baseline image            (previous)
    sc_mask_M0     – (B, 1, *spatial)  SC mask at baseline       (registration anchor)
    sc_mask_M12    – (B, 1, *spatial)  SC mask at follow-up      (registration anchor)
    lesion_mask_M0 – (B, 1, *spatial)  lesion mask at baseline   (temporal context)

One output:
    logits         – (B, n_classes, *spatial)  lesion segmentation at M12

Architecture differences vs. MambaXNet:
    - SEM in_channels: 1 → 3  (sc_mask_M0 ‖ sc_mask_M12 ‖ lesion_mask_M0)
    - All three masks are concatenated before being passed to the SEM, so the
      module learns a joint shape embedding that encodes both the spatial
      alignment cues (SC) and the lesion prior (M0 lesion). The SC masks are
      never part of any prediction target, so no capacity is wasted on SC
      segmentation.

Everything else (encoder, decoder, M-CAM, skip connections) is identical.

Author: Pierre-Louis Benveniste
"""

import json
import pydoc

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

import os
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from mcam import MCAM


# ──────────────────────────────────────────────────────────────────────────────
# Weight loading (identical to original)
# ──────────────────────────────────────────────────────────────────────────────

def load_nnunet_weights(model_folder: str) -> ResidualEncoderUNet:
    """Load a pretrained nnU-Net ResEncoderUNet from a fold_0 checkpoint."""
    checkpoint_path  = f"{model_folder}/fold_0/checkpoint_best.pth"
    json_config_path = f"{model_folder}/plans.json"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("network_weights", checkpoint.get("state_dict", checkpoint))

    with open(json_config_path) as f:
        init_args = json.load(f)
    init_args = init_args["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]

    for key in ("conv_op", "norm_op", "nonlin"):
        if isinstance(init_args.get(key), str):
            init_args[key] = pydoc.locate(init_args[key])

    model = ResidualEncoderUNet(input_channels=1, num_classes=2, **init_args)
    model.load_state_dict(state_dict)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Shape Extractor Module
# ──────────────────────────────────────────────────────────────────────────────

class ShapeExtractorModule(nn.Module):
    """
    Three sequential (Conv3d → ReLU) blocks.

    Args:
        in_channels:  number of input mask channels.
                      For SC-Lesion this is 3:
                        ch0 = sc_mask_M0
                        ch1 = sc_mask_M12
                        ch2 = lesion_mask_M0
        out_channels: latent channel dim (matched to encoder feature channels).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        mid = max(out_channels // 2, in_channels)
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """masks : (B, in_channels, D, H, W)  →  (B, out_channels, D, H, W)"""
        return self.blocks(masks)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class MambaXNetSCLesion(nn.Module):
    """
    MambaXNet extended with explicit SC-mask registration anchors.

    The SEM now processes three mask channels (sc_M0, sc_M12, lesion_M0)
    concatenated along the channel dim.  All other components are unchanged.
    """

    def __init__(self, n_channels: int = 1, resenc_model=None, n_classes: int = 2):
        super().__init__()

        # ── Encoder (from pretrained ResEncoderUNet) ──────────────────────────
        enc = resenc_model.encoder
        self.enc_stem   = enc.stem
        self.enc_stage0 = enc.stages[0]
        self.enc_stage1 = enc.stages[1]
        self.enc_stage2 = enc.stages[2]
        self.enc_stage3 = enc.stages[3]
        self.enc_stage4 = enc.stages[4]
        self.enc_stage5 = enc.stages[5]

        # ── Decoder (from pretrained ResEncoderUNet) ──────────────────────────
        dec = resenc_model.decoder
        self.transpconvs = dec.transpconvs   # ModuleList of 5 ConvTranspose3d
        self.dec_stages  = dec.stages        # ModuleList of 5 StackedConvBlocks
        self.seg_layers  = dec.seg_layers    # ModuleList of 5 Conv3d heads

        # ── SEM: 3 input channels (sc_M0, sc_M12, lesion_M0) ─────────────────
        self.sem = ShapeExtractorModule(in_channels=3, out_channels=32)

        # ── M-CAM blocks at the three finest encoder levels ───────────────────
        self.m_cam1 = MCAM(in_channels=32,  embed_dim=128, num_heads=8, sem_channels=32)
        self.m_cam2 = MCAM(in_channels=64,  embed_dim=64,  num_heads=8, sem_channels=32)
        self.m_cam3 = MCAM(in_channels=128, embed_dim=32,  num_heads=8, sem_channels=32)

    def forward(
        self,
        image_M12:      torch.Tensor,   # (B, 1, *spatial) — follow-up image
        image_M0:       torch.Tensor,   # (B, 1, *spatial) — baseline image
        sc_mask_M0:     torch.Tensor,   # (B, 1, *spatial) — SC mask at M0
        sc_mask_M12:    torch.Tensor,   # (B, 1, *spatial) — SC mask at M12
        lesion_mask_M0: torch.Tensor,   # (B, 1, *spatial) — lesion mask at M0
    ) -> torch.Tensor:
        """Returns (B, n_classes, *spatial) logits for the lesion mask at M12."""

        # ── Encoder features for M12 (current) ───────────────────────────────
        e1 = self.enc_stage0(self.enc_stem(image_M12))
        e2 = self.enc_stage1(e1)
        e3 = self.enc_stage2(e2)
        e4 = self.enc_stage3(e3)
        e5 = self.enc_stage4(e4)
        e6 = self.enc_stage5(e5)

        # ── Encoder features for M0 (previous) — three finest levels only ────
        e1_prev = self.enc_stage0(self.enc_stem(image_M0))
        e2_prev = self.enc_stage1(e1_prev)
        e3_prev = self.enc_stage2(e2_prev)

        # ── Shape embedding from all three mask channels ──────────────────────
        masks   = torch.cat([sc_mask_M0, sc_mask_M12, lesion_mask_M0], dim=1)  # (B, 3, *)
        f_sem   = self.sem(masks)                                                # (B, 32, *)

        # ── M-CAM cross-attention at the three finest encoder resolutions ─────
        e1_mcam = self.m_cam1(e1, e1_prev, f_sem)
        e2_mcam = self.m_cam2(e2, e2_prev, f_sem)
        e3_mcam = self.m_cam3(e3, e3_prev, f_sem)

        # ── Decoder ───────────────────────────────────────────────────────────
        # stage 0: bottleneck e6 → upsample → cat(e5) → 640→320
        d = self.transpconvs[0](e6)
        d = self.dec_stages[0](torch.cat([d, e5], dim=1))
        # stage 1: → cat(e4) → 512→256
        d = self.transpconvs[1](d)
        d = self.dec_stages[1](torch.cat([d, e4], dim=1))
        # stage 2: → cat(e3_mcam) → 256→128
        d = self.transpconvs[2](d)
        d = self.dec_stages[2](torch.cat([d, e3_mcam], dim=1))
        # stage 3: → cat(e2_mcam) → 128→64
        d = self.transpconvs[3](d)
        d = self.dec_stages[3](torch.cat([d, e2_mcam], dim=1))
        # stage 4: → cat(e1_mcam) → 64→32
        d = self.transpconvs[4](d)
        d = self.dec_stages[4](torch.cat([d, e1_mcam], dim=1))

        return self.seg_layers[4](d)   # (B, n_classes, *spatial)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available — running on CPU (slow).")

    model_folder = "/path/to/nnunet/fold"   # update before running
    resenc_model = load_nnunet_weights(model_folder)
    print("Pretrained weights loaded.")

    model = MambaXNetSCLesion(n_channels=1, resenc_model=resenc_model, n_classes=2)
    model.to(device).eval()
    print("MambaXNetSCLesion initialised.")

    shape = (1, 1, 32, 128, 128)
    image_M12      = torch.randn(*shape, device=device)
    image_M0       = torch.randn(*shape, device=device)
    sc_mask_M0     = torch.zeros(*shape, device=device)
    sc_mask_M12    = torch.zeros(*shape, device=device)
    lesion_mask_M0 = torch.zeros(*shape, device=device)

    with torch.no_grad():
        out = model(image_M12, image_M0, sc_mask_M0, sc_mask_M12, lesion_mask_M0)
    print(f"Output shape: {out.shape}")   # expected: (1, 2, 32, 128, 128)


if __name__ == "__main__":
    main()
