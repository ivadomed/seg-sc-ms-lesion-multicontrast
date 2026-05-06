import torch
import torch.nn as nn
from mcam import MCAM
from sem import ShapeExtractorModule
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import json
import pydoc # Useful for importing classes from strings


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


class MambaXNet(nn.Module):
    def __init__(self, n_channels=1, resenc_model=None, n_classes=2):
        super(MambaXNet, self).__init__()
        # Reconstruction of the encoder layers
        self.encoder_layer1 = resenc_model.encoder[-1]
        self.encoder_layer2 = resenc_model.encoder[-2]
        self.encoder_layer3 = resenc_model.encoder[-3]
        self.encoder_layer4 = resenc_model.encoder[-4]
        self.encoder_layer5 = resenc_model.encoder[-5]
        self.encoder_layer6 = resenc_model.encoder[-6]
        # Reconstruction of the decoder layers
        self.decoder_layer1 = resenc_model.decoder[-1]
        self.decoder_layer2 = resenc_model.decoder[-2]
        self.decoder_layer3 = resenc_model.decoder[-3]
        self.decoder_layer4 = resenc_model.decoder[-4]
        self.decoder_layer5 = resenc_model.decoder[-5]
        self.decoder_layer6 = resenc_model.decoder[-6]
        
        # self.sem = ShapeExtractorModule()
        
        # # M-CAM blocks integrated at the last three upsampling levels
        # self.m_cams = nn.ModuleList([
        #     MCAM(channels=256, num_patches=..., embedding_dim=256),
        #     MCAM(channels=128, num_patches=..., embedding_dim=128),
        #     MCAM(channels=64, num_patches=..., embedding_dim=64)
        # ])

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(nn.Conv3d(in_c, out_c, 3, padding=1), nn.ReLU())

    def forward(self, i_t, i_prev, m_prev):
        # 1. Feature Extraction (Shared weights for Enc_t and Enc_prev)
        ## For image at t
        features_t = []
        x_t = i_t
        for layer in self.encoder:
            x_t = layer(x_t)
            features_t.append(x_t)
        ## For image at t-1 (shared encoder)  
        features_prev = []
        x_prev = i_prev
        for i in range(3): # Auxiliary encoder truncated to top 3 feature maps
            x_prev = self.encoder[i](x_prev)
            features_prev.append(x_prev)
            
        # 2. Shape Extraction of previous mask
        f_sem = self.sem(m_prev)
        
        # 3. Decoding with M-CAM Fusion
        ## The features at times t goes directly through th


def main():
    # Try loading the nnU-Net weights into MambaXNet
    checkpoint_path = '/home/plbenveniste/net/longitudinal_mamba/trained_resencUnet/nnUNetTrainerDiceCELoss_noSmooth_4000epochs_fromScratch__nnUNetResEncUNetL1x1x1_Model2_Plans__3d_fullres/fold_0/checkpoint_best.pth'
    json_config_path = '/home/plbenveniste/net/longitudinal_mamba/trained_resencUnet/nnUNetTrainerDiceCELoss_noSmooth_4000epochs_fromScratch__nnUNetResEncUNetL1x1x1_Model2_Plans__3d_fullres/plans.json'
    # Load resenc model with pretrained nnU-Net weights
    resenc_model = load_nnunet_weights(checkpoint_path, json_config_path)
    print("Pretrained nnU-Net weights loaded successfully into ResEncoderUNet model.")

    # initialize MambaXNet with the loaded ResEncoderUNet model
    model = MambaXNet(n_channels=1, resenc_model=resenc_model, n_classes=2)
    print("MambaXNet initialized with pretrained nnU-Net weights.")
    
    
if __name__ == "__main__":
   main()