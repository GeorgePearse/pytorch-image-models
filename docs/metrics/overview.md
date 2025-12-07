# Model Metrics Overview

This section contains comprehensive performance metrics for all models available in `timm`. Metrics are organized by release date for easier navigation.

!!! success "🔥 Complete Benchmark Table"
    **New!** View all model benchmarks in one comprehensive sortable table:

    **[→ Complete Benchmark Table with ALL Models](all-models.md)**

    This table includes 150+ model variants with full performance metrics, making it easy to compare models and find the best option for your use case.

## How to Use These Metrics

All metrics shown are ImageNet-1k validation results unless otherwise specified. Key metrics include:

- **top1**: Top-1 accuracy (%)
- **top5**: Top-5 accuracy (%)
- **param_count**: Number of parameters (millions)
- **img_size**: Input image resolution

## Browse by Year

- [2025 Models](2025-models.md) - Latest model releases and updates
- [2024 Models](2024-models.md) - Models released throughout 2024
- [Historical Models](historical.md) - Earlier releases and classic architectures

## Top Performing Models (2025)

### Vision Transformers (ViT)

| Model | Top-1 | Top-5 | Params (M) | Image Size |
|:------|:-----:|:-----:|:----------:|:----------:|
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 89.42 | - | 400+ | 378 |
| vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k | 87.438 | 98.256 | 64.11 | 384 |
| vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k | 88.1 | - | 150+ | 448 |
| vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k | 87.9 | - | 150+ | 384 |

### MambaOut Models

| Model | Top-1 | Top-5 | Params (M) | Image Size |
|:------|:-----:|:-----:|:----------:|:----------:|
| mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k | 87.506 | 98.428 | 101.66 | 384 |
| mambaout_base_plus_rw.sw_e150_in12k_ft_in1k | 86.912 | 98.236 | 101.66 | 288 |
| mambaout_base_plus_rw.sw_e150_in12k_ft_in1k | 86.632 | 98.156 | 101.66 | 224 |

### MobileNet Family

| Model | Top-1 | Top-5 | Params (M) | Image Size |
|:------|:-----:|:-----:|:----------:|:----------:|
| mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k | 84.99 | 97.294 | 32.59 | 544 |
| mobilenetv4_hybrid_large.ix_e600_r384_in1k | 84.356 | 96.892 | 37.76 | 448 |
| mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k | 84.772 | 97.344 | 32.59 | 480 |

### ResNet Family

| Model | Top-1 | Top-5 | Params (M) | Image Size |
|:------|:-----:|:-----:|:----------:|:----------:|
| resnet50d.ra4_e3600_r224_in1k | 81.838 | 95.922 | 25.58 | 288 |
| resnet50d.ra4_e3600_r224_in1k | 80.952 | 95.384 | 25.58 | 224 |

## Model Categories

### By Architecture Type

- **Vision Transformers (ViT)**: Standard and optimized transformer architectures
- **Hybrid Models**: Combining convolutions and attention mechanisms
- **ConvNets**: Pure convolutional architectures (ResNet, EfficientNet, MobileNet, etc.)
- **Meta-architectures**: PoolFormer, MetaFormer, etc.

### By Size Category

- **Tiny** (< 10M params): Optimized for mobile and edge devices
- **Small** (10-50M params): Balanced accuracy and efficiency
- **Medium** (50-100M params): High accuracy with reasonable compute
- **Large** (100M+ params): Maximum accuracy for server deployment

## Performance vs Efficiency

Different models offer different trade-offs between accuracy and computational efficiency:

- **Mobile/Edge**: MobileNetV4, EfficientNet-B0, FastViT
- **Server/Cloud**: ViT-Large, EVA-02, SigLIP models
- **Balanced**: ResNet-50D, EfficientNet-B1-B3, ViT-Base

## Training Recipes

Many models use specific training recipes:

- **RA4**: RandAugment with 4 augmentation magnitude
- **SBB**: "Searching for Better ViT Baselines" training recipe
- **IN12K**: Pre-trained on ImageNet-12k before fine-tuning on ImageNet-1k
- **WebLI**: Pre-trained on WebLI dataset (large-scale web images)

## Links

- [2025 Model Metrics](2025-models.md)
- [2024 Model Metrics](2024-models.md)
- [Historical Metrics](historical.md)
