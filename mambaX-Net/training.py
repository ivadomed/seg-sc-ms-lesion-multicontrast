"""
This script contains the training loop for the model. It includes functions for training and evaluating the model, as well as saving checkpoints and logging metrics.

Input:
    --unet: path to the pretrained UNet model
    --data: path to the dataset 
    --output: path to the output folder

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from mambaxnet import load_nnunet_weights, MambaXNet
from load_dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for the model.")
    parser.add_argument("--unet", type=str, required=True, help="Path to the pretrained UNet model.")
    parser.add_argument("--data", type=str, required=True, help="Path to the MSD dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    unet_path = args.unet
    data_path = args.data
    output_path = args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load the data from the MSD dataset
    train_loader, val_loader, test_loader = get_dataloaders(json_path=data_path)


if __name__ == "__main__":
    main()