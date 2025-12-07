# 2025 Model Metrics

This page contains all models released or updated in 2025.

## July 2025: Naver ROPE-ViT Models

ROPE-ViT models from Naver with rotary position embeddings. [Paper](https://github.com/naver-ai/rope-vit)

| Model | Img Size | Top-1 | Top-5 | Params (M) |
|:------|:--------:|:-----:|:-----:|:----------:|
| vit_large_patch16_rope_mixed_ape_224.naver_in1k | 224 | 84.840 | 97.122 | 304.40 |
| vit_large_patch16_rope_mixed_224.naver_in1k | 224 | 84.828 | 97.116 | 304.20 |
| vit_large_patch16_rope_ape_224.naver_in1k | 224 | 84.650 | 97.154 | 304.37 |
| vit_large_patch16_rope_224.naver_in1k | 224 | 84.648 | 97.122 | 304.17 |
| vit_base_patch16_rope_mixed_ape_224.naver_in1k | 224 | 83.894 | 96.754 | 86.59 |
| vit_base_patch16_rope_mixed_224.naver_in1k | 224 | 83.804 | 96.712 | 86.44 |
| vit_base_patch16_rope_ape_224.naver_in1k | 224 | 83.782 | 96.610 | 86.59 |
| vit_base_patch16_rope_224.naver_in1k | 224 | 83.718 | 96.672 | 86.43 |
| vit_small_patch16_rope_224.naver_in1k | 224 | 81.230 | 95.022 | 21.98 |
| vit_small_patch16_rope_mixed_224.naver_in1k | 224 | 81.216 | 95.022 | 21.99 |
| vit_small_patch16_rope_ape_224.naver_in1k | 224 | 81.004 | 95.016 | 22.06 |
| vit_small_patch16_rope_mixed_ape_224.naver_in1k | 224 | 80.986 | 94.976 | 22.06 |

## June 2025: NaFlexViT Models

Initial NaFlexViT checkpoints with native aspect ratio support.

| Model | Top-1 | Top-5 | Params (M) | Eval Seq Len |
|:------|:-----:|:-----:|:----------:|:------------:|
| naflexvit_base_patch16_par_gap.e300_s576_in1k | 83.67 | 96.45 | 86.63 | 576 |
| naflexvit_base_patch16_parfac_gap.e300_s576_in1k | 83.63 | 96.41 | 86.46 | 576 |
| naflexvit_base_patch16_gap.e300_s576_in1k | 83.50 | 96.46 | 86.63 | 576 |

## May 2025: Searching for Better ViT Baselines

Exploring model shapes between Tiny and Base with SBB training recipes.

### ImageNet-12k Pretrain + ImageNet-1k Fine-tune

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k | 86.202 | 97.874 | 64.11 | 256 |
| vit_betwixt_patch16_reg4_gap_256.sbb_in12k_ft_in1k | 85.418 | 97.480 | 60.40 | 256 |

### ImageNet-1k Only Training

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k | 84.322 | 96.812 | 63.95 | 256 |
| vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k | 83.906 | 96.684 | 60.23 | 256 |
| vit_base_patch16_rope_reg1_gap_256.sbb_in1k | 83.866 | 96.670 | 86.43 | 256 |
| vit_medium_patch16_rope_reg1_gap_256.sbb_in1k | 83.810 | 96.824 | 38.74 | 256 |
| vit_betwixt_patch16_reg4_gap_256.sbb_in1k | 83.706 | 96.616 | 60.40 | 256 |
| vit_betwixt_patch16_reg1_gap_256.sbb_in1k | 83.628 | 96.544 | 60.40 | 256 |
| vit_medium_patch16_reg4_gap_256.sbb_in1k | 83.470 | 96.622 | 38.88 | 256 |
| vit_medium_patch16_reg1_gap_256.sbb_in1k | 83.462 | 96.548 | 38.88 | 256 |
| vit_little_patch16_reg4_gap_256.sbb_in1k | 82.514 | 96.262 | 22.52 | 256 |
| vit_wee_patch16_reg1_gap_256.sbb_in1k | 80.256 | 95.360 | 13.42 | 256 |
| vit_pwee_patch16_reg1_gap_256.sbb_in1k | 80.072 | 95.136 | 15.25 | 256 |

## February 2025: SigLIP 2 & SO150M2 Models

### SigLIP 2 ViT Image Encoders

Variable resolution / aspect NaFlex versions available.

### SO150M2 ViT Models

SBB-trained models with excellent ImageNet results:

| Model | Top-1 | Img Size | Params (M) |
|:------|:-----:|:--------:|:----------:|
| vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k | 88.1 | 448 | 150+ |
| vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k | 87.9 | 384 | 150+ |
| vit_so150m2_patch16_reg1_gap_256.sbb_e200_in12k_ft_in1k | 87.3 | 256 | 150+ |
| vit_so150m2_patch16_reg4_gap_256.sbb_e200_in12k | - | 256 | 150+ |

## January 2025: SO150M ViT Models

| Model | Top-1 | Img Size | Params (M) |
|:------|:-----:|:--------:|:----------:|
| vit_so150m_patch16_reg4_gap_384.sbb_e250_in12k_ft_in1k | 87.4 | 384 | 150+ |
| vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k_ft_in1k | 86.7 | 256 | 150+ |
| vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k | - | 256 | 150+ |

## Model Naming Convention

Understanding the model names:

- **vit**: Vision Transformer base architecture
- **patch16**: 16x16 patch size
- **rope**: Rotary Position Embedding
- **reg1/reg4**: Number of register tokens
- **gap**: Global Average Pooling
- **256/384/448**: Input image resolution
- **sbb**: Searching for Better Baselines training recipe
- **e200/e250**: Number of training epochs
- **in12k**: Pre-trained on ImageNet-12k
- **ft_in1k**: Fine-tuned on ImageNet-1k
- **naver_in1k**: Trained by Naver on ImageNet-1k

## Training Details

### SBB Recipe
The "Searching for Better ViT Baselines" recipe focuses on:
- Efficient training for GPU-constrained environments
- Strong regularization with register tokens
- Optimal hyperparameters for medium-sized models

### ROPE Variants
Rotary Position Embedding (ROPE) variants include:
- **rope**: Standard ROPE
- **rope_mixed**: Mixed ROPE mode
- **ape**: Absolute Position Embedding addition
- **mixed_ape**: Combined mixed ROPE with APE

## Links

- [Back to Overview](overview.md)
- [2024 Models](2024-models.md)
- [Historical Models](historical.md)
