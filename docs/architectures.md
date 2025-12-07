# Model Architectures

This page provides an overview of all model architectures available in `timm`.

## Vision Transformers (ViT)

Transformer-based architectures for computer vision.

### Standard ViT
- **Vision Transformer** - https://arxiv.org/abs/2010.11929
- Pure transformer architecture with patch embeddings
- Available in Tiny, Small, Base, Large, Huge sizes

### ViT Variants
- **DeiT** - Data-efficient Image Transformers - https://arxiv.org/abs/2012.12877
- **DeiT-III** - https://arxiv.org/pdf/2204.07118.pdf
- **BEiT** - BERT Pre-Training for Image Transformers - https://arxiv.org/abs/2106.08254
- **BEiT-V2** - https://arxiv.org/abs/2208.06366
- **BEiT3** - https://arxiv.org/abs/2208.10442
- **CaiT** - Class-Attention in Image Transformers - https://arxiv.org/abs/2103.17239
- **FlexiViT** - https://arxiv.org/abs/2212.08013
- **ROPE-ViT** - https://arxiv.org/abs/2403.13298
- **NaFlexViT** - Native aspect ratio, variable resolution support
- **PE (Perception Encoder)** - https://arxiv.org/abs/2504.13181

### CLIP Vision Encoders
- **SigLIP** (image encoder) - https://arxiv.org/abs/2303.15343
- **SigLIP 2** (image encoder) - https://arxiv.org/abs/2502.14786
- **MobileCLIP** - https://arxiv.org/abs/2311.17049
- **ViTamin** - https://arxiv.org/abs/2404.02132
- **TinyCLIP** - Compact CLIP vision towers

## Hierarchical Transformers

Transformer architectures with hierarchical feature maps.

### Swin Transformer Family
- **Swin Transformer** - https://arxiv.org/abs/2103.14030
- **Swin Transformer V2** - https://arxiv.org/abs/2111.09883
- **Swin S3 (AutoFormerV2)** - https://arxiv.org/abs/2111.14725

### Other Hierarchical
- **Twins** - Spatial Attention in Vision Transformers - https://arxiv.org/pdf/2104.13840.pdf
- **Focal Net** - Focal Modulation Networks - https://arxiv.org/abs/2203.11926
- **GCViT** - Global Context Vision Transformer - https://arxiv.org/abs/2206.09959
- **MaxViT** - Multi-Axis Vision Transformer - https://arxiv.org/abs/2204.01697
- **CoaT** - Co-Scale Conv-Attentional Image Transformers - https://arxiv.org/abs/2104.06399

## Hybrid Conv-Transformer Models

Combining convolutional and transformer components.

### Major Hybrids
- **CoAtNet** - Convolution and Attention - https://arxiv.org/abs/2106.04803
- **ConViT** - Soft Convolutional Inductive Biases - https://arxiv.org/abs/2103.10697
- **LeViT** - Vision Transformer in ConvNet's Clothing - https://arxiv.org/abs/2104.01136
- **MobileViT** - https://arxiv.org/abs/2110.02178
- **MobileViT-V2** - https://arxiv.org/abs/2206.02680
- **EfficientViT (MIT)** - https://arxiv.org/abs/2205.14756
- **EfficientViT (MSRA)** - https://arxiv.org/abs/2305.07027
- **FastViT** - https://arxiv.org/abs/2303.14189
- **Next-ViT** - https://arxiv.org/abs/2207.05501

## Modern ConvNets

Pure convolutional architectures with modern design principles.

### ConvNeXt Family
- **ConvNeXt** - https://arxiv.org/abs/2201.03545
- **ConvNeXt-V2** - http://arxiv.org/abs/2301.00808
- Modern ConvNet design inspired by transformers

### Recent ConvNets
- **MambaOut** - https://arxiv.org/abs/2405.07992
- **FasterNet** - https://arxiv.org/abs/2303.03667
- **SwiftFormer** - https://arxiv.org/pdf/2303.15446
- **StarNet** - https://arxiv.org/abs/2403.19967
- **RDNet** - DenseNets Reloaded - https://arxiv.org/abs/2403.19588
- **EdgeNeXt** - https://arxiv.org/abs/2206.10589
- **InceptionNeXt** - https://arxiv.org/abs/2303.16900

## Mobile & Efficient Models

Architectures optimized for mobile and edge deployment.

### MobileNet Family
- **MobileNet-V1** - Depthwise separable convolutions
- **MobileNet-V2** - https://arxiv.org/abs/1801.04381
- **MobileNet-V3** - https://arxiv.org/abs/1905.02244
- **MobileNet-V4** - https://arxiv.org/abs/2404.10518
- **MobileNet-V5** - Backbone for Gemma 3n

### EfficientNet Family
- **EfficientNet (B0-B7)** - https://arxiv.org/abs/1905.11946
- **EfficientNet-EdgeTPU** - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
- **EfficientNet V2** - https://arxiv.org/abs/2104.00298
- **EfficientNet NoisyStudent** - https://arxiv.org/abs/1911.04252
- **EfficientNet AdvProp** - https://arxiv.org/abs/1911.09665

### Other Mobile Models
- **GhostNet** - https://arxiv.org/abs/1911.11907
- **GhostNet-V2** - https://arxiv.org/abs/2211.12905
- **GhostNet-V3** - https://arxiv.org/abs/2404.11202
- **RepGhostNet** - https://arxiv.org/abs/2211.06088
- **MobileOne** - https://arxiv.org/abs/2206.04040
- **RepViT** - https://arxiv.org/abs/2307.09283
- **TinyViT** - https://arxiv.org/abs/2207.10666
- **SHViT** - https://arxiv.org/abs/2401.16456
- **MNASNet** - https://arxiv.org/abs/1807.11626
- **FBNet-C** - https://arxiv.org/abs/1812.03443
- **MixNet** - https://arxiv.org/abs/1907.09595
- **TinyNet** - https://arxiv.org/abs/2010.14819
- **Single-Path NAS** - https://arxiv.org/abs/1904.02877
- **HardCoRe-NAS** - https://arxiv.org/abs/2102.11646
- **LCNet** - https://arxiv.org/abs/2109.15099

## ResNet & Variants

Classic and improved residual network architectures.

### Standard ResNet
- **ResNet (v1b/v1.5)** - https://arxiv.org/abs/1512.03385
- **ResNeXt** - https://arxiv.org/abs/1611.05431
- **Big Transfer ResNetV2 (BiT)** - https://arxiv.org/abs/1912.11370

### ResNet Improvements
- **ResNet-D** - Bag of Tricks - https://arxiv.org/abs/1812.01187
- **ResNet-RS** - https://arxiv.org/abs/2103.07579
- **ECA-ResNet** - https://arxiv.org/abs/1910.03151v4
- **SE-ResNet** - Squeeze-and-Excitation - https://arxiv.org/abs/1709.01507
- **SK-ResNet** - Selective Kernel - https://arxiv.org/abs/1903.06586
- **Res2Net** - https://arxiv.org/abs/1904.01169
- **ResNeSt** - https://arxiv.org/abs/2004.08955

### Pre-trained ResNet Variants
- **WSL** - Weakly-supervised Instagram pretrained - https://arxiv.org/abs/1805.00932
- **SSL/SWSL** - Semi-supervised/Semi-weakly supervised - https://arxiv.org/abs/1905.00546

## Attention & Meta-Former Models

Models based on attention mechanisms or meta-architectures.

### Pure Attention
- **Bottleneck Transformers** - https://arxiv.org/abs/2101.11605
- **Lambda Networks** - https://arxiv.org/abs/2102.08602
- **Halo Nets** - https://arxiv.org/abs/2103.12731

### Meta-Formers
- **MetaFormer** - PoolFormer-v2, ConvFormer, CAFormer - https://arxiv.org/abs/2210.13452
- **PoolFormer** - https://arxiv.org/abs/2111.11418

### MLP-Based
- **MLP-Mixer** - https://arxiv.org/abs/2105.01601
- **ResMLP** - https://arxiv.org/abs/2105.03404
- **gMLP** - https://arxiv.org/abs/2105.08050
- **Sequencer2D** - https://arxiv.org/abs/2205.01972

## Specialized Architectures

### Multi-Scale & Pyramids
- **PVT-V2** - Improved Pyramid Vision Transformer - https://arxiv.org/abs/2106.13797
- **PiT** - Pooling-based Vision Transformer - https://arxiv.org/abs/2103.16302
- **MViT-V2** - Improved Multiscale Vision Transformer - https://arxiv.org/abs/2112.01526
- **NesT** - Aggregating Nested Transformers - https://arxiv.org/abs/2105.12723
- **TNT** - Transformer-iN-Transformer - https://arxiv.org/abs/2103.00112

### EVA Models
- **EVA** - https://arxiv.org/abs/2211.07636
- **EVA-02** - https://arxiv.org/abs/2303.11331

### DaViT & Hiera
- **DaViT** - Dual Attention Vision Transformers
- **Hiera** - Hierarchical vision transformer from Meta - https://github.com/facebookresearch/hiera

## Classic & Legacy Architectures

Time-tested architectures still widely used.

### Dense & Efficient
- **DenseNet** - https://arxiv.org/abs/1608.06993
- **DPN** - Dual-Path Network - https://arxiv.org/abs/1707.01629
- **GPU-Efficient Networks** - https://arxiv.org/abs/2006.14090

### Inception Family
- **Inception-V3** - https://arxiv.org/abs/1512.00567
- **Inception-ResNet-V2 / Inception-V4** - https://arxiv.org/abs/1602.07261
- **Xception** - https://arxiv.org/abs/1610.02357
- **Xception (Modified Aligned)** - https://arxiv.org/abs/1802.02611

### Detection Backbones
- **DLA** - Deep Layer Aggregation - https://arxiv.org/abs/1707.06484
- **HRNet** - High-Resolution Net - https://arxiv.org/abs/1908.07919
- **CSPNet** - Cross-Stage Partial Networks - https://arxiv.org/abs/1911.11929
- **TResNet** - https://arxiv.org/abs/2003.13630
- **VovNet V2 and V1** - https://arxiv.org/abs/1911.06667

### Specialized
- **SelecSLS** - https://arxiv.org/abs/1907.00837
- **ReXNet** - https://arxiv.org/abs/2007.00992
- **ResMLP** - https://arxiv.org/abs/2105.03404
- **RepVGG** - https://arxiv.org/abs/2101.03697
- **SqueezeNet** - Lightweight architecture
- **VGG** - https://arxiv.org/abs/1409.1556

### NAS Models
- **NASNet-A** - https://arxiv.org/abs/1707.07012
- **PNasNet** - https://arxiv.org/abs/1712.00559
- **EfficientNet (AutoML)** - https://arxiv.org/abs/1905.11946

### Normalizer-Free
- **NFNet-F** - https://arxiv.org/abs/2102.06171
- **NF-RegNet / NF-ResNet** - https://arxiv.org/abs/2101.08692

### Other Notable
- **RegNet** - https://arxiv.org/abs/2003.13678
- **RegNetZ** - https://arxiv.org/abs/2103.06877
- **VOLO** - Vision Outlooker - https://arxiv.org/abs/2106.13112
- **XCiT** - Cross-Covariance Image Transformers - https://arxiv.org/abs/2106.09681
- **Visformer** - https://arxiv.org/abs/2104.12533
- **HGNet / HGNet-V2** - From PaddlePaddle

## Architecture Categories

### By Paradigm
1. **Pure Transformers**: ViT, DeiT, BEiT
2. **Hierarchical Transformers**: Swin, PVT, Twins
3. **Hybrid**: CoAtNet, LeViT, MobileViT
4. **Pure ConvNets**: ResNet, EfficientNet, ConvNeXt
5. **MLP-Based**: MLP-Mixer, ResMLP, gMLP
6. **Meta-Architectures**: MetaFormer, PoolFormer

### By Use Case
- **Mobile/Edge**: MobileNet series, GhostNet, RepViT
- **Server/Cloud**: Large ViT, EVA, Swin-Large
- **Balanced**: EfficientNet, ResNet-50, ViT-Base
- **Research**: Experimental architectures, NAS models

### By Training Type
- **Supervised**: Standard ImageNet training
- **Self-Supervised**: BEiT, DINO variants
- **Weakly-Supervised**: WSL ResNets
- **Distillation**: DeiT models
- **CLIP-style**: SigLIP, MobileCLIP

## Architecture Selection Guide

Choose based on your constraints:

- **Accuracy Priority**: Large ViT, EVA, Swin-Large
- **Speed Priority**: MobileNet-V4, EfficientNet-B0, FastViT
- **Memory Priority**: Tiny models, MobileNet variants
- **Balanced**: ResNet-50D, EfficientNet-B1-B3, ViT-Base
- **Novel Features**: NaFlexViT (variable aspect), ROPE-ViT (rotary embeddings)

## Links

- [Model Metrics](metrics/overview.md)
- [Getting Started](getting-started.md)
- [GitHub Repository](https://github.com/GeorgePearse/pytorch-image-models)
