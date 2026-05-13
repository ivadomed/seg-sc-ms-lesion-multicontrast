# Longitudinal lesion segmentation with MambaX-Net

This folder explores longitudinal lesion segmentation using a MambaX-Net architecture: paper https://arxiv.org/abs/2510.17529.

Installation used for the following server:
```console
Ubuntu: 20.04
CUDA version: 12.0
GLIBC: 2.31
```

Commands:

```console
# Create conda env
conda create -n venv_mamba python=3.13
conda activate venv_mamba

# Install torch
pip install torch==2.5.1

# Install mamba
git clone git@github.com:state-spaces/mamba.git
cd mamba
git checkout v2.3.1
pip install -e .  --no-build-isolation

# Install dynamic-network-architectures for ResEncUnet loading
pip install dynamic-network-architectures --no-deps
```