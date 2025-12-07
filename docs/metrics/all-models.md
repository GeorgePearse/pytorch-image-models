# Complete Model Benchmark Results

This page contains a comprehensive table of all model benchmarks extracted from timm. Use your browser's search (Ctrl+F / Cmd+F) or the table sorting to find specific models.

!!! tip "How to Use This Table"
    - **Click column headers** to sort by that metric
    - **Use browser search** (Ctrl+F / Cmd+F) to find specific models
    - **Top-1 and Top-5** are ImageNet-1k validation accuracies (%)
    - **Params** are in millions (M)
    - **Image Size** is the input resolution used for validation

## Complete Benchmark Table

### 2025 Models

#### Naver ROPE-ViT Models (July 2025)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| vit_large_patch16_rope_mixed_ape_224.naver_in1k | 2025 | 84.840 | 97.122 | 304.40 | 224 | ROPE + Mixed + APE |
| vit_large_patch16_rope_mixed_224.naver_in1k | 2025 | 84.828 | 97.116 | 304.20 | 224 | ROPE + Mixed |
| vit_large_patch16_rope_ape_224.naver_in1k | 2025 | 84.650 | 97.154 | 304.37 | 224 | ROPE + APE |
| vit_large_patch16_rope_224.naver_in1k | 2025 | 84.648 | 97.122 | 304.17 | 224 | ROPE |
| vit_base_patch16_rope_mixed_ape_224.naver_in1k | 2025 | 83.894 | 96.754 | 86.59 | 224 | ROPE + Mixed + APE |
| vit_base_patch16_rope_mixed_224.naver_in1k | 2025 | 83.804 | 96.712 | 86.44 | 224 | ROPE + Mixed |
| vit_base_patch16_rope_ape_224.naver_in1k | 2025 | 83.782 | 96.610 | 86.59 | 224 | ROPE + APE |
| vit_base_patch16_rope_224.naver_in1k | 2025 | 83.718 | 96.672 | 86.43 | 224 | ROPE |
| vit_small_patch16_rope_224.naver_in1k | 2025 | 81.230 | 95.022 | 21.98 | 224 | ROPE |
| vit_small_patch16_rope_mixed_224.naver_in1k | 2025 | 81.216 | 95.022 | 21.99 | 224 | ROPE + Mixed |
| vit_small_patch16_rope_ape_224.naver_in1k | 2025 | 81.004 | 95.016 | 22.06 | 224 | ROPE + APE |
| vit_small_patch16_rope_mixed_ape_224.naver_in1k | 2025 | 80.986 | 94.976 | 22.06 | 224 | ROPE + Mixed + APE |

#### NaFlexViT Models (June 2025)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Eval Seq Len | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------------:|:------|
| naflexvit_base_patch16_par_gap.e300_s576_in1k | 2025 | 83.67 | 96.45 | 86.63 | 224 | 576 | Native aspect |
| naflexvit_base_patch16_parfac_gap.e300_s576_in1k | 2025 | 83.63 | 96.41 | 86.46 | 224 | 576 | Factorized pos embed |
| naflexvit_base_patch16_gap.e300_s576_in1k | 2025 | 83.50 | 96.46 | 86.63 | 224 | 576 | Native aspect |

#### SBB ViT Models (May 2025)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k | 2025 | 86.202 | 97.874 | 64.11 | 256 | IN12k pretrain |
| vit_betwixt_patch16_reg4_gap_256.sbb_in12k_ft_in1k | 2025 | 85.418 | 97.480 | 60.40 | 256 | IN12k pretrain |
| vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k | 2025 | 84.322 | 96.812 | 63.95 | 256 | ROPE |
| vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k | 2025 | 83.906 | 96.684 | 60.23 | 256 | ROPE |
| vit_base_patch16_rope_reg1_gap_256.sbb_in1k | 2025 | 83.866 | 96.670 | 86.43 | 256 | ROPE |
| vit_medium_patch16_rope_reg1_gap_256.sbb_in1k | 2025 | 83.810 | 96.824 | 38.74 | 256 | ROPE |
| vit_betwixt_patch16_reg4_gap_256.sbb_in1k | 2025 | 83.706 | 96.616 | 60.40 | 256 | SBB recipe |
| vit_betwixt_patch16_reg1_gap_256.sbb_in1k | 2025 | 83.628 | 96.544 | 60.40 | 256 | SBB recipe |
| vit_medium_patch16_reg4_gap_256.sbb_in1k | 2025 | 83.470 | 96.622 | 38.88 | 256 | SBB recipe |
| vit_medium_patch16_reg1_gap_256.sbb_in1k | 2025 | 83.462 | 96.548 | 38.88 | 256 | SBB recipe |
| vit_little_patch16_reg4_gap_256.sbb_in1k | 2025 | 82.514 | 96.262 | 22.52 | 256 | SBB recipe |
| vit_wee_patch16_reg1_gap_256.sbb_in1k | 2025 | 80.256 | 95.360 | 13.42 | 256 | SBB recipe |
| vit_pwee_patch16_reg1_gap_256.sbb_in1k | 2025 | 80.072 | 95.136 | 15.25 | 256 | SBB recipe |

#### SO150M2 ViT Models (February 2025)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k | 2025 | 88.1 | - | 150+ | 448 | IN12k pretrain |
| vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k | 2025 | 87.9 | - | 150+ | 384 | IN12k pretrain |
| vit_so150m2_patch16_reg1_gap_256.sbb_e200_in12k_ft_in1k | 2025 | 87.3 | - | 150+ | 256 | IN12k pretrain |

#### SO150M ViT Models (January 2025)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| vit_so150m_patch16_reg4_gap_384.sbb_e250_in12k_ft_in1k | 2025 | 87.4 | - | 150+ | 384 | IN12k pretrain |
| vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k_ft_in1k | 2025 | 86.7 | - | 150+ | 256 | IN12k pretrain |

### 2024 Models

#### MambaOut Models (October 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k | 2024 | 87.506 | 98.428 | 101.66 | 384 | IN12k pretrain |
| mambaout_base_plus_rw.sw_e150_in12k_ft_in1k | 2024 | 86.912 | 98.236 | 101.66 | 288 | IN12k pretrain |
| mambaout_base_plus_rw.sw_e150_in12k_ft_in1k | 2024 | 86.632 | 98.156 | 101.66 | 224 | IN12k pretrain |
| mambaout_base_tall_rw.sw_e500_in1k | 2024 | 84.974 | 97.332 | 86.48 | 288 | Tall variant |
| mambaout_base_wide_rw.sw_e500_in1k | 2024 | 84.962 | 97.208 | 94.45 | 288 | Wide variant |
| mambaout_base_short_rw.sw_e500_in1k | 2024 | 84.832 | 97.270 | 88.83 | 288 | Short variant |
| mambaout_base.in1k | 2024 | 84.720 | 96.930 | 84.81 | 288 | Base |
| mambaout_small_rw.sw_e450_in1k | 2024 | 84.598 | 97.098 | 48.50 | 288 | Small variant |
| mambaout_small.in1k | 2024 | 84.500 | 96.974 | 48.49 | 288 | Small |
| mambaout_base_wide_rw.sw_e500_in1k | 2024 | 84.454 | 96.864 | 94.45 | 224 | Wide variant |
| mambaout_base_tall_rw.sw_e500_in1k | 2024 | 84.434 | 96.958 | 86.48 | 224 | Tall variant |
| mambaout_base_short_rw.sw_e500_in1k | 2024 | 84.362 | 96.952 | 88.83 | 224 | Short variant |
| mambaout_base.in1k | 2024 | 84.168 | 96.680 | 84.81 | 224 | Base |
| mambaout_small.in1k | 2024 | 84.086 | 96.630 | 48.49 | 224 | Small |
| mambaout_small_rw.sw_e450_in1k | 2024 | 84.024 | 96.752 | 48.50 | 224 | Small variant |
| mambaout_tiny.in1k | 2024 | 83.448 | 96.538 | 26.55 | 288 | Tiny |
| mambaout_tiny.in1k | 2024 | 82.736 | 96.100 | 26.55 | 224 | Tiny |
| mambaout_kobe.in1k | 2024 | 81.054 | 95.718 | 9.14 | 288 | Ultra-small |
| mambaout_kobe.in1k | 2024 | 79.986 | 94.986 | 9.14 | 224 | Ultra-small |
| mambaout_femto.in1k | 2024 | 79.848 | 95.140 | 7.30 | 288 | Ultra-small |
| mambaout_femto.in1k | 2024 | 78.870 | 94.408 | 7.30 | 224 | Ultra-small |

#### SigLIP SO400M Models (October 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 2024 | 89.42 | - | 400+ | 378 | WebLI pretrain |
| vit_so400m_patch14_siglip_gap_378.webli_ft_in1k | 2024 | 89.03 | - | 400+ | 378 | WebLI pretrain + GAP |

#### ConvNeXt Zepto Models (October 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| convnext_zepto_rms_ols.ra4_e3600_r224_in1k | 2024 | 73.20 | - | 2.2 | 224 | Overlapped stem |
| convnext_zepto_rms.ra4_e3600_r224_in1k | 2024 | 72.81 | - | 2.2 | 224 | Patch stem |

#### Updated SBB ViT Models (August 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k | 2024 | 87.438 | 98.256 | 64.11 | 384 | IN12k pretrain |
| vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k | 2024 | 86.608 | 97.934 | 64.11 | 256 | IN12k pretrain |
| vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k | 2024 | 86.594 | 98.020 | 60.40 | 384 | IN12k pretrain |
| vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k | 2024 | 85.734 | 97.610 | 60.40 | 256 | IN12k pretrain |

#### Baseline Challenge Models (August 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| resnet50d.ra4_e3600_r224_in1k | 2024 | 81.838 | 95.922 | 25.58 | 288 | RA4 recipe |
| resnet50d.ra4_e3600_r224_in1k | 2024 | 80.952 | 95.384 | 25.58 | 224 | RA4 recipe |
| efficientnet_b1.ra4_e3600_r240_in1k | 2024 | 81.440 | 95.700 | 7.79 | 288 | RA4 recipe |
| efficientnet_b1.ra4_e3600_r240_in1k | 2024 | 80.406 | 95.152 | 7.79 | 240 | RA4 recipe |
| mobilenetv1_125.ra4_e3600_r224_in1k | 2024 | 77.600 | 93.804 | 6.27 | 256 | RA4 recipe |
| mobilenetv1_125.ra4_e3600_r224_in1k | 2024 | 76.924 | 93.234 | 6.27 | 224 | RA4 recipe |

#### Hiera Models (August 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k | 2024 | 84.912 | 97.260 | 35.01 | 256 | IN12k pretrain |
| hiera_small_abswin_256.sbb2_pd_e200_in12k_ft_in1k | 2024 | 84.560 | 97.106 | 35.01 | 256 | IN12k pretrain + PD |

#### MobileNetV4 Models (June-July 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k | 2024 | 84.990 | 97.294 | 32.59 | 544 | Anti-aliased + IN12k |
| mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k | 2024 | 84.772 | 97.344 | 32.59 | 480 | Anti-aliased + IN12k |
| mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k | 2024 | 84.640 | 97.114 | 32.59 | 448 | Anti-aliased + IN12k |
| mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k | 2024 | 84.314 | 97.102 | 32.59 | 384 | Anti-aliased + IN12k |
| mobilenetv4_hybrid_large.ix_e600_r384_in1k | 2024 | 84.356 | 96.892 | 37.76 | 448 | Hybrid + IX init |
| mobilenetv4_hybrid_large.ix_e600_r384_in1k | 2024 | 83.990 | 96.702 | 37.76 | 384 | Hybrid + IX init |
| mobilenetv4_conv_aa_large.e600_r384_in1k | 2024 | 83.824 | 96.734 | 32.59 | 480 | Anti-aliased |
| mobilenetv4_hybrid_medium.ix_e550_r384_in1k | 2024 | 83.394 | 96.760 | 11.07 | 448 | Hybrid + IX init |
| mobilenetv4_conv_large.e600_r384_in1k | 2024 | 83.392 | 96.622 | 32.59 | 448 | Conv variant |
| mobilenetv4_conv_aa_large.e600_r384_in1k | 2024 | 83.244 | 96.392 | 32.59 | 384 | Anti-aliased |
| mobilenetv4_hybrid_medium.ix_e550_r384_in1k | 2024 | 82.968 | 96.474 | 11.07 | 384 | Hybrid + IX init |
| mobilenetv4_conv_large.e600_r384_in1k | 2024 | 82.952 | 96.266 | 32.59 | 384 | Conv variant |
| mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k | 2024 | 82.990 | 96.670 | 11.07 | 320 | IN12k pretrain |
| mobilenetv4_conv_large.e500_r256_in1k | 2024 | 82.674 | 96.310 | 32.59 | 320 | Conv variant |
| mobilenetv4_hybrid_medium.ix_e550_r256_in1k | 2024 | 82.492 | 96.278 | 11.07 | 320 | Hybrid + IX init |
| mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k | 2024 | 82.364 | 96.256 | 11.07 | 256 | IN12k pretrain |
| mobilenetv4_conv_large.e500_r256_in1k | 2024 | 81.862 | 95.690 | 32.59 | 256 | Conv variant |
| mobilenetv4_hybrid_medium.ix_e550_r256_in1k | 2024 | 81.446 | 95.704 | 11.07 | 256 | Hybrid + IX init |
| mobilenetv4_hybrid_medium.e500_r224_in1k | 2024 | 81.276 | 95.742 | 11.07 | 256 | Hybrid |
| mobilenetv4_conv_medium.e500_r256_in1k | 2024 | 80.858 | 95.768 | 9.72 | 320 | Conv medium |
| mobilenetv4_hybrid_medium.e500_r224_in1k | 2024 | 80.442 | 95.380 | 11.07 | 224 | Hybrid |
| mobilenetv4_conv_blur_medium.e500_r224_in1k | 2024 | 80.142 | 95.298 | 9.72 | 256 | Blur pooling |
| mobilenetv4_conv_medium.e500_r256_in1k | 2024 | 79.928 | 95.184 | 9.72 | 256 | Conv medium |
| mobilenetv4_conv_medium.e500_r224_in1k | 2024 | 79.808 | 95.186 | 9.72 | 256 | Conv medium |
| mobilenetv4_conv_blur_medium.e500_r224_in1k | 2024 | 79.438 | 94.932 | 9.72 | 224 | Blur pooling |
| mobilenetv4_conv_medium.e500_r224_in1k | 2024 | 79.094 | 94.770 | 9.72 | 224 | Conv medium |
| mobilenet_edgetpu_v2_m.ra4_e3600_r224_in1k | 2024 | 80.700 | - | 6.9 | 256 | EdgeTPU optimized |
| mobilenet_edgetpu_v2_m.ra4_e3600_r224_in1k | 2024 | 80.100 | - | 6.9 | 224 | EdgeTPU optimized |
| mobilenetv4_conv_small.e2400_r224_in1k | 2024 | 74.616 | 92.072 | 3.77 | 256 | Conv small |
| mobilenetv4_conv_small.e1200_r224_in1k | 2024 | 74.292 | 92.116 | 3.77 | 256 | Conv small |
| mobilenetv4_conv_small.e2400_r224_in1k | 2024 | 73.756 | 91.422 | 3.77 | 224 | Conv small |
| mobilenetv4_conv_small.e1200_r224_in1k | 2024 | 73.454 | 91.340 | 3.77 | 224 | Conv small |

#### MobileNet Baseline Models (July 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| efficientnet_b0.ra4_e3600_r224_in1k | 2024 | 79.364 | 94.754 | 5.29 | 256 | RA4 recipe |
| efficientnet_b0.ra4_e3600_r224_in1k | 2024 | 78.584 | 94.338 | 5.29 | 224 | RA4 recipe |
| mobilenetv1_100h.ra4_e3600_r224_in1k | 2024 | 76.596 | 93.272 | 5.28 | 256 | RA4 recipe |
| mobilenetv1_100.ra4_e3600_r224_in1k | 2024 | 76.094 | 93.004 | 4.23 | 256 | RA4 recipe |
| mobilenetv1_100h.ra4_e3600_r224_in1k | 2024 | 75.662 | 92.504 | 5.28 | 224 | RA4 recipe |
| mobilenetv1_100.ra4_e3600_r224_in1k | 2024 | 75.382 | 92.312 | 4.23 | 224 | RA4 recipe |

#### MobileNetV3 & Small Models (September 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| mobilenetv3_large_150d.ra4_e3600_r256_in1k | 2024 | 81.81 | - | 7.5 | 320 | RA4 recipe |
| mobilenetv3_large_150d.ra4_e3600_r256_in1k | 2024 | 80.94 | - | 7.5 | 256 | RA4 recipe |
| mobilenetv3_large_100.ra4_e3600_r224_in1k | 2024 | 77.16 | - | 5.5 | 256 | RA4 recipe |
| mobilenetv3_large_100.ra4_e3600_r224_in1k | 2024 | 76.31 | - | 5.5 | 224 | RA4 recipe |
| mobilenetv4_conv_small_050.e3000_r224_in1k | 2024 | 65.81 | - | 1.9 | 256 | 0.5x width |
| mobilenetv4_conv_small_050.e3000_r224_in1k | 2024 | 64.76 | - | 1.9 | 224 | 0.5x width |

#### Tiny Test Models (September 2024)

| Model | Year | Top-1 | Top-5 | Params (M) | Img Size | Notes |
|:------|:----:|:-----:|:-----:|:----------:|:--------:|:------|
| test_efficientnet.r160_in1k | 2024 | 47.156 | 71.726 | 0.36 | 192 | Test model |
| test_byobnet.r160_in1k | 2024 | 46.698 | 71.674 | 0.46 | 192 | Test model |
| test_efficientnet.r160_in1k | 2024 | 46.426 | 70.928 | 0.36 | 160 | Test model |
| test_byobnet.r160_in1k | 2024 | 45.378 | 70.572 | 0.46 | 160 | Test model |
| test_vit.r160_in1k | 2024 | 42.000 | 68.664 | 0.37 | 192 | Test model |
| test_vit.r160_in1k | 2024 | 40.822 | 67.212 | 0.37 | 160 | Test model |

## Summary Statistics

### Top-10 Models by Accuracy (Top-1)

1. **vit_so400m_patch14_siglip_378.webli_ft_in1k**: 89.42% (400M params, 378px)
2. **vit_so400m_patch14_siglip_gap_378.webli_ft_in1k**: 89.03% (400M params, 378px)
3. **vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k**: 88.1% (150M params, 448px)
4. **vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k**: 87.9% (150M params, 384px)
5. **mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k**: 87.506% (101.66M params, 384px)
6. **vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k**: 87.438% (64.11M params, 384px)
7. **vit_so150m_patch16_reg4_gap_384.sbb_e250_in12k_ft_in1k**: 87.4% (150M params, 384px)
8. **vit_so150m2_patch16_reg1_gap_256.sbb_e200_in12k_ft_in1k**: 87.3% (150M params, 256px)
9. **mambaout_base_plus_rw.sw_e150_in12k_ft_in1k**: 86.912% (101.66M params, 288px)
10. **vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k_ft_in1k**: 86.7% (150M params, 256px)

### Most Efficient Models (Params < 10M)

1. **mambaout_kobe.in1k**: 81.054% with 9.14M params @ 288px
2. **mobilenetv4_conv_medium**: 80.858% with 9.72M params @ 320px
3. **mambaout_kobe.in1k**: 79.986% with 9.14M params @ 224px
4. **mambaout_femto.in1k**: 79.848% with 7.30M params @ 288px
5. **efficientnet_b0.ra4_e3600_r224_in1k**: 79.364% with 5.29M params @ 256px

### Architecture Distribution

- **Vision Transformers (ViT)**: 40+ variants
- **MambaOut**: 21 variants
- **MobileNetV4**: 30+ variants
- **ConvNeXt**: 2 variants (Zepto)
- **Hiera**: 2 variants
- **ResNet**: 2 variants with modern training
- **EfficientNet**: 3 variants with modern training
- **MobileNet V1/V3**: 8 variants

## Notes

- All metrics are on ImageNet-1k validation set unless otherwise specified
- **IN12k pretrain**: Model was pre-trained on ImageNet-12k before fine-tuning on ImageNet-1k
- **WebLI**: Pre-trained on large-scale web images
- **RA4**: RandAugment with magnitude 4 training recipe
- **SBB**: "Searching for Better ViT Baselines" training recipe
- **ROPE**: Rotary Position Embedding
- **GAP**: Global Average Pooling
- **Anti-aliased**: Uses blur pooling for better shift-invariance

## Links

- [Back to Metrics Overview](overview.md)
- [2025 Models Details](2025-models.md)
- [2024 Models Details](2024-models.md)
- [Historical Models](historical.md)
