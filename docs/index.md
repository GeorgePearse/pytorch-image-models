# PyTorch Image Models (timm)

!!! info "Welcome to timm Documentation"
    Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

## Introduction

The work of many others is present here. All source material is acknowledged via links to github, arxiv papers, etc in the README, documentation, and code docstrings.

## Key Features

- **1000+ Pre-trained Models**: Wide variety of state-of-the-art architectures
- **Consistent API**: All models share common interfaces for feature extraction and classifier access
- **Multi-scale Feature Extraction**: Easy feature pyramid extraction from any model
- **Production Ready**: High-performance training and inference scripts
- **Extensive Augmentations**: Modern augmentation techniques including RandAugment, AutoAugment, CutMix, and more
- **Modern Optimizers**: Including LION, AdamW, LAMB, LARS, and many more

## Quick Links

- 🔥 **[Complete Benchmark Table](metrics/all-models.md)** - All 150+ models in one sortable table
- [Model Metrics](metrics/overview.md) - Browse performance metrics by year
- [Getting Started](getting-started.md) - Installation and basic usage
- [Official Documentation](https://huggingface.co/docs/hub/timm)
- [GitHub Repository](https://github.com/GeorgePearse/pytorch-image-models)
- [Papers with Code](https://paperswithcode.com/lib/timm)

## What's New

### December 2025
- Lightweight task abstraction added
- Logits and feature distillation support added to train script
- Removed old APEX AMP support

### November 2025
- Fixed LayerScale init bug
- EfficientNet-X and EfficientNet-H B5 model weights added
- Muon optimizer implementation added

### October 2025
- DINOv3 ConvNeXt and ViT models added
- MobileCLIP-2 vision encoders added
- MetaCLIP-2 Worldwide ViT encoder weights added
- SigLIP-2 NaFlex ViT encoder weights added

## Installation

```bash
pip install timm
```

For development installation:

```bash
git clone https://github.com/GeorgePearse/pytorch-image-models
cd pytorch-image-models
pip install -e .
```

## Quick Start

```python
import timm
import torch

# List all available models
model_names = timm.list_models()

# Create a model with pretrained weights
model = timm.create_model('resnet50', pretrained=True)
model.eval()

# Prepare an image (3x224x224)
x = torch.randn(1, 3, 224, 224)

# Get predictions
output = model(x)
```

## Citation

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

## License

The code is licensed Apache 2.0. See [Licenses](https://github.com/GeorgePearse/pytorch-image-models#licenses) for details on pretrained weights.
