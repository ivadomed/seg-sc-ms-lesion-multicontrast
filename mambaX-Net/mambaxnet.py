import torch
import torch.nn as nn
# from mcam import MCAM
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
    def __init__(self, in_channels=1, out_channels=32):
        super(ShapeExtractorModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, m_prev):
        return self.layers(m_prev)


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

        # Reconstruction of the decoder layers
        decoder = resenc_model.decoder.encoder
        self.dec_stem = decoder.stem
        self.dec_stage0 = decoder.stages[0]
        self.dec_stage1 = decoder.stages[1]
        self.dec_stage2 = decoder.stages[2]
        self.dec_stage3 = decoder.stages[3]
        self.dec_stage4 = decoder.stages[4]
        self.dec_stage5 = decoder.stages[5]

        # Build the SEM module to extract shape features from the previous time point
        self.sem = ShapeExtractorModule(in_channels=1, out_channels=32)
        
        # # M-CAM blocks integrated at the last three upsampling levels
        # self.m_cams = nn.ModuleList([
        #     MCAM(channels=256, num_patches=..., embedding_dim=256),
        #     MCAM(channels=128, num_patches=..., embedding_dim=128),
        #     MCAM(channels=64, num_patches=..., embedding_dim=64)
        # ])

    def forward(self, i_t, i_prev, m_prev):
        # Build features output by encoder layers for time t
        e1 = self.enc_stage0(self.enc_stem(i_t))
        e2 = self.enc_stage1(e1)
        e3 = self.enc_stage2(e2)
        e4 = self.enc_stage3(e3)
        e5 = self.enc_stage4(e4)
        e6 = self.enc_stage5(e5)
        # Build features output by encoder layers for time t-1
        e1_prev = self.enc_stage0(self.enc_stem(i_prev))
        e2_prev = self.enc_stage1(e1_prev)
        e3_prev = self.enc_stage2(e2_prev)

        # Extract shape features from the previous time point using SEM
        shape_features = self.sem(m_prev)

        # # Merge with M-CAM blocks at the last three upsampling levels
        # e1_mcam = self.m_cams(e1, e1_prev, shape_features)
        # e2_mcam = self.m_cams(e2, e2_prev, shape_features)
        # e3_mcam = self.m_cams(e3, e3_prev, shape_features)

        # # Build outputs of the decoder layers, integrating the M-CAM outputs
        # d1 = self.decoder[0](e6) # First decoder layer processes the bottleneck features
        # d2 = self.decoder[1](d1 + e5)
        # d3 = self.decoder[2](d2 + e4)
        # d4 = self.decoder[3](d3 + e3_mcam)
        # d5 = self.decoder[4](d4 + e2_mcam)
        # d6 = self.decoder[5](d5 + e1_mcam)

        return shape_features


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

    # Print show the size of inputs and outputs of the ResEncoderUNet to verify correct loading
    # dummy_input = torch.randn(1, 1, 128, 128, 128) # Example input shape
    # with torch.no_grad():
    #     resenc_output = resenc_model(dummy_input)
    # print("ResEncoderUNet forward pass successful. Output shape:", resenc_output.shape)

    # initialize MambaXNet with the loaded ResEncoderUNet model
    model = MambaXNet(n_channels=1, resenc_model=resenc_model, n_classes=2)
    model.to(device)
    model.eval()
    print("MambaXNet initialized with pretrained nnU-Net weights.")

    # Generate a random input tensor to test the forward pass
    i_t = torch.randn(1, 1, 64, 64, 64).to(device) # Example input for time t
    i_prev = torch.randn(1, 1, 64, 64, 64).to(device) # Example input for time t-1
    m_prev = torch.randn(1, 1, 64, 64, 64).to(device) # Example mask from time t-1
    output = model(i_t, i_prev, m_prev)
    print("MambaXNet forward pass successful. Output shape:", output.shape)    
    
    
if __name__ == "__main__":
   main()