import torch
import torch.nn as nn
import torch.nn.functional as F
from mcam import MCAM
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import json
import pydoc # Useful for importing classes from strings
from mamba_ssm import Mamba # Requires mamba-ssm installation


def load_nnunet_weights(checkpoint_path, json_config_path):
    """
    This function loads the weights of the pretrained ResEncoderUNet and returns the model.
    """
    # Load the nnU-Net checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract the state_dict from the checkpoint
    if 'network_weights' in checkpoint:
        state_dict = checkpoint['network_weights']
    else:
        # Fallback if the checkpoint was saved differently
        state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Load the json config file to get the initialization arguments for the nnU-Net model
    with open(json_config_path, 'r') as f:
        init_args = json.load(f)
    init_args = init_args['configurations']['3d_fullres']['architecture']['arch_kwargs']
    # Convert strings to actual objects/classes
    # These specific keys are strings in the JSON but must be objects for the class init
    if isinstance(init_args.get('conv_op'), str):
        init_args['conv_op'] = pydoc.locate(init_args['conv_op'])
    if isinstance(init_args.get('norm_op'), str):
        init_args['norm_op'] = pydoc.locate(init_args['norm_op'])
    if isinstance(init_args.get('nonlin'), str):
        init_args['nonlin'] = pydoc.locate(init_args['nonlin'])
    
    # Build a ResidualEncoderUNet to load the weights and then extract the relevant parts for MambaXNet
    res_enc_unet = ResidualEncoderUNet(input_channels=1, num_classes=2, **init_args)
    res_enc_unet.load_state_dict(state_dict)

    return res_enc_unet


class ShapeExtractorModule(nn.Module):
    """
    SEM: three sequential (Conv3d + ReLU) blocks.

    Args:
        in_channels:  number of mask channels (e.g. 1 for lesion masks)
        out_channels: latent channel dim (set to match encoder feature channels)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
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

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask : [B, C_mask, D, H, W]
        return: [B, out_channels, D, H, W]
        """
        return self.blocks(mask)


class MambaXNet(nn.Module):
    def __init__(self, n_channels=1, resenc_model=None, n_classes=2):
        super(MambaXNet, self).__init__()
        # Reconstruction of the encoder layers
        enc = resenc_model.encoder
        self.enc_stem = enc.stem
        self.enc_stage0 = enc.stages[0]
        self.enc_stage1 = enc.stages[1]
        self.enc_stage2 = enc.stages[2]
        self.enc_stage3 = enc.stages[3]
        self.enc_stage4 = enc.stages[4]
        self.enc_stage5 = enc.stages[5]

        # Decoder layers — pulled from the actual decoder, not decoder.encoder
        dec = resenc_model.decoder
        self.transpconvs = dec.transpconvs   # ModuleList of 5 ConvTranspose3d
        self.dec_stages   = dec.stages        # ModuleList of 5 StackedConvBlocks
        self.seg_layers   = dec.seg_layers    # ModuleList of 5 Conv3d heads

        # Build the SEM module to extract shape features from the previous time point
        self.sem = ShapeExtractorModule(in_channels=1, out_channels=32)
        
        # M-CAM blocks integrated at the last three upsampling levels
        self.m_cam1 = MCAM(in_channels=32, embed_dim=128, num_heads=8, sem_channels=32)
        self.m_cam2 = MCAM(in_channels=64, embed_dim=64, num_heads=8, sem_channels=32)
        self.m_cam3 = MCAM(in_channels=128, embed_dim=32, num_heads=8, sem_channels=32)

    def forward(self, i_t, i_prev, m_prev):
        # Build features output by encoder layers for time t
        e1 = self.enc_stage0(self.enc_stem(i_t))
        print("e1.shape:", e1.shape)
        e2 = self.enc_stage1(e1)
        print("e2.shape:", e2.shape)
        e3 = self.enc_stage2(e2)
        print("e3.shape:", e3.shape)
        e4 = self.enc_stage3(e3)
        print("e4.shape:", e4.shape)
        e5 = self.enc_stage4(e4)
        print("e5.shape:", e5.shape)
        e6 = self.enc_stage5(e5)
        print("e6.shape:", e6.shape)
        
        # Build features output by encoder layers for time t-1
        e1_prev = self.enc_stage0(self.enc_stem(i_prev))
        print("e1_prev.shape:", e1_prev.shape)
        e2_prev = self.enc_stage1(e1_prev)
        print("e2_prev.shape:", e2_prev.shape)
        e3_prev = self.enc_stage2(e2_prev)
        print("e3_prev.shape:", e3_prev.shape)

        # Extract shape features from the previous time point using SEM
        m_prev_shape = self.sem(m_prev)
        print("m_prev_shape.shape:", m_prev_shape.shape)

        # # Merge with M-CAM blocks at the last three upsampling levels
        e1_mcam = self.m_cam1(e1, e1_prev, m_prev_shape)
        e2_mcam = self.m_cam2(e2, e2_prev, m_prev_shape)
        e3_mcam = self.m_cam3(e3, e3_prev, m_prev_shape)

        # --- Decoder (transpconv → cat with skip → stage) ---
        # stage 0: bottleneck e6 → upsample → cat(e5) → 640→320
        d = self.transpconvs[0](e6)            # [B, 320, ...]
        d = self.dec_stages[0](torch.cat([d, e5], dim=1))   # [B, 320, ...]
        # stage 1: → cat(e4) → 512→256
        d = self.transpconvs[1](d)
        d = self.dec_stages[1](torch.cat([d, e4], dim=1))   # [B, 256, ...]
        # stage 2: → cat(e3_mcam) → 256→128
        d = self.transpconvs[2](d)
        d = self.dec_stages[2](torch.cat([d, e3_mcam], dim=1))  # [B, 128, ...]
        # stage 3: → cat(e2_mcam) → 128→64
        d = self.transpconvs[3](d)
        d = self.dec_stages[3](torch.cat([d, e2_mcam], dim=1))  # [B, 64, ...]
        # stage 4: → cat(e1_mcam) → 64→32
        d = self.transpconvs[4](d)
        d = self.dec_stages[4](torch.cat([d, e1_mcam], dim=1))  # [B, 32, ...]
        # Final segmentation head
        out = self.seg_layers[4](d)   # [B, 2, D, H, W]
        
        return e1_mcam


def main():
    # Use CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check it torch had access to cuda
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation and GPU configuration.")
        return
    else:
        print("CUDA is available. Proceeding with GPU computations.")

    # Try loading the nnU-Net weights into MambaXNet
    model_folder = '/home/plbenveniste/net/longitudinal_mamba/trained_resencUnet/nnUNetTrainerDiceCELoss_noSmooth_4000epochs_fromScratch__nnUNetResEncUNetL1x1x1_Model2_Plans__3d_fullres'
    checkpoint_path = f'{model_folder}/fold_0/checkpoint_best.pth'
    json_config_path = f'{model_folder}/plans.json'
    # Load resenc model with pretrained nnU-Net weights
    resenc_model = load_nnunet_weights(checkpoint_path, json_config_path)
    print("Pretrained nnU-Net weights loaded successfully into ResEncoderUNet model.")

    # initialize MambaXNet with the loaded ResEncoderUNet model
    model = MambaXNet(n_channels=1, resenc_model=resenc_model, n_classes=2)
    model.to(device)
    model.eval()
    print("MambaXNet initialized with pretrained nnU-Net weights.")

    # Generate a random input tensor to test the forward pass
    i_t = torch.randn(1, 1, 32, 128, 128).to(device) # Example input for time t
    i_prev = torch.randn(1, 1, 32, 128, 128).to(device) # Example input for time t-1
    m_prev = torch.randn(1, 1, 32, 128, 128).to(device) # Example mask from time t-1
    output = model(i_t, i_prev, m_prev)
    print("MambaXNet forward pass successful. Output shape:", output.shape)    
    
    
if __name__ == "__main__":
   main()