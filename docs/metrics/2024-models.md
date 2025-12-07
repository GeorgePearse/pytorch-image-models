# 2024 Model Metrics

This page contains all models released or updated in 2024.

## October 2024: MambaOut Models

MambaOut models - ConvNeXt-style architecture with gating, no SSM. [Paper](https://arxiv.org/abs/2405.07992)

| Model | Img Size | Top-1 | Top-5 | Params (M) |
|:------|:--------:|:-----:|:-----:|:----------:|
| mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k | 384 | 87.506 | 98.428 | 101.66 |
| mambaout_base_plus_rw.sw_e150_in12k_ft_in1k | 288 | 86.912 | 98.236 | 101.66 |
| mambaout_base_plus_rw.sw_e150_in12k_ft_in1k | 224 | 86.632 | 98.156 | 101.66 |
| mambaout_base_tall_rw.sw_e500_in1k | 288 | 84.974 | 97.332 | 86.48 |
| mambaout_base_wide_rw.sw_e500_in1k | 288 | 84.962 | 97.208 | 94.45 |
| mambaout_base_short_rw.sw_e500_in1k | 288 | 84.832 | 97.270 | 88.83 |
| mambaout_base.in1k | 288 | 84.720 | 96.930 | 84.81 |
| mambaout_small_rw.sw_e450_in1k | 288 | 84.598 | 97.098 | 48.50 |
| mambaout_small.in1k | 288 | 84.500 | 96.974 | 48.49 |
| mambaout_base_wide_rw.sw_e500_in1k | 224 | 84.454 | 96.864 | 94.45 |
| mambaout_base_tall_rw.sw_e500_in1k | 224 | 84.434 | 96.958 | 86.48 |
| mambaout_base_short_rw.sw_e500_in1k | 224 | 84.362 | 96.952 | 88.83 |
| mambaout_base.in1k | 224 | 84.168 | 96.680 | 84.81 |
| mambaout_small.in1k | 224 | 84.086 | 96.630 | 48.49 |
| mambaout_small_rw.sw_e450_in1k | 224 | 84.024 | 96.752 | 48.50 |
| mambaout_tiny.in1k | 288 | 83.448 | 96.538 | 26.55 |
| mambaout_tiny.in1k | 224 | 82.736 | 96.100 | 26.55 |
| mambaout_kobe.in1k | 288 | 81.054 | 95.718 | 9.14 |
| mambaout_kobe.in1k | 224 | 79.986 | 94.986 | 9.14 |
| mambaout_femto.in1k | 288 | 79.848 | 95.140 | 7.30 |
| mambaout_femto.in1k | 224 | 78.870 | 94.408 | 7.30 |

### SigLIP SO400M ViT Models

| Model | Top-1 | Img Size | Params (M) |
|:------|:-----:|:--------:|:----------:|
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 89.42 | 378 | 400+ |
| vit_so400m_patch14_siglip_gap_378.webli_ft_in1k | 89.03 | 378 | 400+ |
| vit_so400m_patch14_siglip_256.webli (i18n) | - | 256 | 400+ |

### ConvNeXt Zepto Models

Ultra-small ConvNeXt models with RMSNorm (2.2M parameters):

| Model | Top-1 | Img Size | Params (M) |
|:------|:-----:|:--------:|:----------:|
| convnext_zepto_rms_ols.ra4_e3600_r224_in1k | 73.20 | 224 | 2.2 |
| convnext_zepto_rms.ra4_e3600_r224_in1k | 72.81 | 224 | 2.2 |

## September 2024: Tiny Test Models & MobileNet

### MobileNetV4-Conv-Small (0.5x)

| Model | Top-1 (256) | Top-1 (224) | Params (M) |
|:------|:-----------:|:-----------:|:----------:|
| mobilenetv4_conv_small_050.e3000_r224_in1k | 65.81 | 64.76 | 1.9 |

### MobileNetV3-Large Variants (MNV4 Recipe)

| Model | Top-1 (320) | Top-1 (256) | Params (M) |
|:------|:-----------:|:-----------:|:----------:|
| mobilenetv3_large_150d.ra4_e3600_r256_in1k | 81.81 | 80.94 | 7.5 |
| mobilenetv3_large_100.ra4_e3600_r224_in1k | 77.16 | 76.31 | 5.5 |

### Tiny Test Models (< 0.5M params)

For testing and ultra-low resource applications:

| Model | Top-1 (192) | Top-1 (160) | Params (M) |
|:------|:-----------:|:-----------:|:----------:|
| test_efficientnet.r160_in1k | 47.156 | 46.426 | 0.36 |
| test_byobnet.r160_in1k | 46.698 | 45.378 | 0.46 |
| test_vit.r160_in1k | 42.000 | 40.822 | 0.37 |

## August 2024: SBB ViT Updates & Training Challenges

### Updated SBB ViT Models (ImageNet-12k → ImageNet-1k)

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k | 87.438 | 98.256 | 64.11 | 384 |
| vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k | 86.608 | 97.934 | 64.11 | 256 |
| vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k | 86.594 | 98.020 | 60.40 | 384 |
| vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k | 85.734 | 97.610 | 60.40 | 256 |

### MobileNet V4 Baseline Challenges

Training challenges for classic architectures with modern recipes:

| Model | Top-1 (288) | Top-1 (224) | Params (M) |
|:------|:-----------:|:-----------:|:----------:|
| resnet50d.ra4_e3600_r224_in1k | 81.838 | 80.952 | 25.58 |
| efficientnet_b1.ra4_e3600_r240_in1k | 81.440 | 80.406 | 7.79 |
| mobilenetv1_125.ra4_e3600_r224_in1k | 77.600 | 76.924 | 6.27 |

### Hiera Small Models

| Model | Top-1 | Top-5 | Params (M) |
|:------|:-----:|:-----:|:----------:|
| hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k | 84.912 | 97.260 | 35.01 |
| hiera_small_abswin_256.sbb2_pd_e200_in12k_ft_in1k | 84.560 | 97.106 | 35.01 |

## July 2024: MobileNet-V4 & Baseline Models

### MobileNetV4 Models (ImageNet-12k Pretrain)

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k | 84.990 | 97.294 | 32.59 | 544 |
| mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k | 84.772 | 97.344 | 32.59 | 480 |
| mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k | 84.640 | 97.114 | 32.59 | 448 |
| mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k | 84.314 | 97.102 | 32.59 | 384 |
| mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k | 82.990 | 96.670 | 11.07 | 320 |
| mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k | 82.364 | 96.256 | 11.07 | 256 |

### MobileNet-V1 & EfficientNet-B0 Baseline Challenges

Impressive results with modern training recipes:

| Model | Top-1 (256) | Top-1 (224) | Params (M) |
|:------|:-----------:|:-----------:|:----------:|
| efficientnet_b0.ra4_e3600_r224_in1k | 79.364 | 78.584 | 5.29 |
| mobilenetv1_100h.ra4_e3600_r224_in1k | 76.596 | 75.662 | 5.28 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 76.094 | 75.382 | 4.23 |

### MobileEdge TPU V2

| Model | Top-1 (256) | Top-1 (224) | Params (M) |
|:------|:-----------:|:-----------:|:----------:|
| mobilenet_edgetpu_v2_m.ra4_e3600_r224_in1k | 80.700 | 80.100 | 6.9 |

## June 2024: MobileNetV4 Initial Release

### MobileNetV4 Hybrid Models

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| mobilenetv4_hybrid_large.ix_e600_r384_in1k | 84.356 | 96.892 | 37.76 | 448 |
| mobilenetv4_hybrid_large.ix_e600_r384_in1k | 83.990 | 96.702 | 37.76 | 384 |
| mobilenetv4_hybrid_large.e600_r384_in1k | 84.266 | 96.936 | 37.76 | 448 |
| mobilenetv4_hybrid_large.e600_r384_in1k | 83.800 | 96.770 | 37.76 | 384 |
| mobilenetv4_hybrid_medium.ix_e550_r384_in1k | 83.394 | 96.760 | 11.07 | 448 |
| mobilenetv4_hybrid_medium.ix_e550_r384_in1k | 82.968 | 96.474 | 11.07 | 384 |
| mobilenetv4_hybrid_medium.ix_e550_r256_in1k | 82.492 | 96.278 | 11.07 | 320 |
| mobilenetv4_hybrid_medium.ix_e550_r256_in1k | 81.446 | 95.704 | 11.07 | 256 |
| mobilenetv4_hybrid_medium.e500_r224_in1k | 81.276 | 95.742 | 11.07 | 256 |
| mobilenetv4_hybrid_medium.e500_r224_in1k | 80.442 | 95.380 | 11.07 | 224 |

### MobileNetV4 Conv Models

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| mobilenetv4_conv_large.e600_r384_in1k | 83.392 | 96.622 | 32.59 | 448 |
| mobilenetv4_conv_large.e600_r384_in1k | 82.952 | 96.266 | 32.59 | 384 |
| mobilenetv4_conv_large.e500_r256_in1k | 82.674 | 96.310 | 32.59 | 320 |
| mobilenetv4_conv_large.e500_r256_in1k | 81.862 | 95.690 | 32.59 | 256 |
| mobilenetv4_conv_aa_large.e600_r384_in1k | 83.824 | 96.734 | 32.59 | 480 |
| mobilenetv4_conv_aa_large.e600_r384_in1k | 83.244 | 96.392 | 32.59 | 384 |
| mobilenetv4_conv_medium.e500_r256_in1k | 80.858 | 95.768 | 9.72 | 320 |
| mobilenetv4_conv_medium.e500_r256_in1k | 79.928 | 95.184 | 9.72 | 256 |
| mobilenetv4_conv_medium.e500_r224_in1k | 79.808 | 95.186 | 9.72 | 256 |
| mobilenetv4_conv_medium.e500_r224_in1k | 79.094 | 94.770 | 9.72 | 224 |
| mobilenetv4_conv_blur_medium.e500_r224_in1k | 80.142 | 95.298 | 9.72 | 256 |
| mobilenetv4_conv_blur_medium.e500_r224_in1k | 79.438 | 94.932 | 9.72 | 224 |
| mobilenetv4_conv_small.e2400_r224_in1k | 74.616 | 92.072 | 3.77 | 256 |
| mobilenetv4_conv_small.e2400_r224_in1k | 73.756 | 91.422 | 3.77 | 224 |
| mobilenetv4_conv_small.e1200_r224_in1k | 74.292 | 92.116 | 3.77 | 256 |
| mobilenetv4_conv_small.e1200_r224_in1k | 73.454 | 91.340 | 3.77 | 224 |

## Model Naming Convention

- **ra4**: RandAugment training recipe with magnitude 4
- **sw**: StochDepth + weight decay variant
- **e<num>**: Number of training epochs
- **r<size>**: Training resolution
- **in1k**: ImageNet-1k dataset
- **in12k**: ImageNet-12k dataset
- **ft**: Fine-tuned
- **aa**: Anti-aliased downsampling
- **ix**: Improved attention initialization
- **rw**: Ross Wightman (timm author) trained weights

## Links

- [Back to Overview](overview.md)
- [2025 Models](2025-models.md)
- [Historical Models](historical.md)
