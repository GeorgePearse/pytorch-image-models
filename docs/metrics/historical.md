# Historical Model Metrics

This page contains models and metrics from earlier releases (pre-2024).

!!! note "Historical Archive"
    This page archives model performance metrics from the early development of timm. Many of these models remain highly relevant and widely used in production today.

## Classic Vision Transformers

The original Vision Transformer implementations that established the baseline for transformer-based vision models.

### Standard ViT Models

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| vit_base_patch16_224 | 81.8 | 95.6 | 86.6 | 224 |
| vit_base_patch16_384 | 83.1 | 96.5 | 86.6 | 384 |
| vit_large_patch16_224 | 82.6 | 96.1 | 304.3 | 224 |
| vit_large_patch16_384 | 83.8 | 96.8 | 304.3 | 384 |

## ResNet Family

The foundational convolutional architectures that dominated computer vision before transformers.

### Standard ResNet

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| resnet18 | 69.8 | 89.1 | 11.7 | 224 |
| resnet34 | 73.3 | 91.4 | 21.8 | 224 |
| resnet50 | 76.1 | 92.9 | 25.6 | 224 |
| resnet101 | 77.4 | 93.5 | 44.5 | 224 |
| resnet152 | 78.3 | 94.1 | 60.2 | 224 |

### ResNet-D Variants

Improved stem and downsampling:

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| resnet50d | 77.6 | 25.6 | 224 |
| resnet101d | 78.8 | 44.5 | 224 |
| resnet152d | 79.5 | 60.2 | 224 |

## EfficientNet Family

The original compound scaling models that achieved state-of-the-art efficiency.

### EfficientNet-B Series

| Model | Top-1 | Top-5 | Params (M) | Img Size |
|:------|:-----:|:-----:|:----------:|:--------:|
| efficientnet_b0 | 77.7 | 93.5 | 5.3 | 224 |
| efficientnet_b1 | 79.2 | 94.5 | 7.8 | 240 |
| efficientnet_b2 | 80.3 | 95.1 | 9.1 | 260 |
| efficientnet_b3 | 81.7 | 95.9 | 12.2 | 300 |
| efficientnet_b4 | 83.0 | 96.4 | 19.3 | 380 |
| efficientnet_b5 | 83.8 | 96.8 | 30.4 | 456 |
| efficientnet_b6 | 84.1 | 96.9 | 43.0 | 528 |
| efficientnet_b7 | 84.4 | 97.1 | 66.3 | 600 |

### EfficientNet-V2

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| efficientnetv2_s | 83.9 | 21.5 | 384 |
| efficientnetv2_m | 85.1 | 54.1 | 480 |
| efficientnetv2_l | 85.7 | 119.5 | 480 |
| efficientnetv2_xl | 86.2 | 208.1 | 512 |

## MobileNet Family

Efficient architectures designed for mobile and edge devices.

### MobileNet-V2

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| mobilenetv2_050 | 65.4 | 2.0 | 224 |
| mobilenetv2_100 | 72.0 | 3.5 | 224 |
| mobilenetv2_140 | 75.0 | 6.1 | 224 |

### MobileNet-V3

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| mobilenetv3_small_050 | 57.9 | 1.5 | 224 |
| mobilenetv3_small_075 | 65.7 | 2.0 | 224 |
| mobilenetv3_small_100 | 67.7 | 2.5 | 224 |
| mobilenetv3_large_075 | 73.4 | 4.0 | 224 |
| mobilenetv3_large_100 | 75.2 | 5.5 | 224 |

## DeiT (Data-efficient Image Transformers)

Knowledge distillation for Vision Transformers.

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| deit_tiny_patch16_224 | 72.2 | 5.7 | 224 |
| deit_small_patch16_224 | 79.8 | 22.1 | 224 |
| deit_base_patch16_224 | 81.8 | 86.6 | 224 |
| deit_base_patch16_384 | 83.1 | 86.6 | 384 |

## Swin Transformer

Hierarchical Vision Transformer with shifted windows.

### Swin-V1

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| swin_tiny_patch4_window7_224 | 81.2 | 28.3 | 224 |
| swin_small_patch4_window7_224 | 83.2 | 49.6 | 224 |
| swin_base_patch4_window7_224 | 83.5 | 87.8 | 224 |
| swin_base_patch4_window12_384 | 84.5 | 87.9 | 384 |
| swin_large_patch4_window7_224 | 86.3 | 196.5 | 224 |
| swin_large_patch4_window12_384 | 87.3 | 196.7 | 384 |

### Swin-V2

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| swinv2_tiny_window8_256 | 81.8 | 28.3 | 256 |
| swinv2_small_window8_256 | 83.7 | 49.7 | 256 |
| swinv2_base_window8_256 | 84.2 | 87.9 | 256 |
| swinv2_base_window16_256 | 84.6 | 87.9 | 256 |

## ConvNeXt

Modern pure ConvNet architecture inspired by transformers.

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| convnext_tiny | 82.1 | 28.6 | 224 |
| convnext_small | 83.1 | 50.2 | 224 |
| convnext_base | 83.8 | 88.6 | 224 |
| convnext_large | 84.3 | 197.8 | 224 |
| convnext_xlarge | 84.6 | 350.2 | 224 |

## BEiT (BERT Pre-Training for Image Transformers)

Self-supervised pre-trained Vision Transformers.

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| beit_base_patch16_224 | 85.2 | 86.5 | 224 |
| beit_base_patch16_384 | 86.8 | 86.7 | 384 |
| beit_large_patch16_224 | 87.5 | 304.4 | 224 |
| beit_large_patch16_384 | 88.6 | 305.0 | 384 |

## RegNet

Fast and efficient networks from Facebook AI Research.

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| regnetx_002 | 68.8 | 2.7 | 224 |
| regnetx_004 | 72.4 | 5.2 | 224 |
| regnetx_008 | 75.5 | 7.3 | 224 |
| regnetx_016 | 77.0 | 9.2 | 224 |
| regnetx_032 | 78.4 | 15.3 | 224 |
| regnety_002 | 70.3 | 3.2 | 224 |
| regnety_004 | 74.0 | 4.3 | 224 |
| regnety_008 | 76.3 | 6.3 | 224 |
| regnety_016 | 77.9 | 11.2 | 224 |
| regnety_032 | 79.5 | 19.4 | 224 |

## NFNet (Normalizer-Free Networks)

High-performance networks without batch normalization.

| Model | Top-1 | Params (M) | Img Size |
|:------|:-----:|:----------:|:--------:|
| nfnet_f0 | 83.6 | 71.5 | 256 |
| nfnet_f1 | 84.7 | 132.6 | 320 |
| nfnet_f2 | 85.1 | 193.8 | 352 |
| nfnet_f3 | 85.7 | 254.9 | 416 |
| nfnet_f4 | 85.9 | 316.1 | 512 |

## Notable Training Techniques

### Augmentation Strategies

- **AutoAugment**: Learned augmentation policies
- **RandAugment**: Simplified random augmentation
- **TrivialAugment**: Minimalist augmentation
- **CutMix**: Cutting and mixing training images
- **Mixup**: Linearly interpolating images and labels

### Regularization Methods

- **DropPath**: Stochastic depth
- **DropBlock**: Structured dropout
- **MixUp**: Label smoothing via mixing
- **CutMix**: Spatial mixing

### Training Recipes

- **A3**: Advanced augmentation recipe from ResNet-RS
- **RA**: RandAugment-based training
- **DeiT**: Knowledge distillation from teacher models

## Legacy But Still Relevant

Many of these "historical" models remain highly relevant:

- **ResNet-50** is still the standard baseline for many tasks
- **EfficientNet** models offer excellent efficiency for production
- **MobileNet** variants are widely deployed on mobile devices
- **ViT-Base** remains a common transformer baseline

## Evolution Timeline

```
2015 ──── ResNet
2017 ──── MobileNet-V1, SENet
2018 ──── MobileNet-V2, EfficientNet
2019 ──── EfficientNet scaling, MobileNet-V3
2020 ──── Vision Transformer (ViT), DeiT
2021 ──── Swin, BEiT, ConvNeXt
2022 ──── ConvNeXt-V2, EfficientNet-V2
2023 ──── Modern training recipes, SBB baselines
2024 ──── MobileNet-V4, MambaOut, improved recipes
2025 ──── ROPE-ViT, NaFlexViT, SO150M variants
```

## Links

- [Back to Overview](overview.md)
- [2025 Models](2025-models.md)
- [2024 Models](2024-models.md)
