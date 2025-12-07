# Training Scripts

The `timm` repository includes reference training, validation, and inference scripts for ImageNet and other datasets.

## Overview

The training scripts are located in the root of the repository and support:

- **Multi-GPU training** with DistributedDataParallel
- **Mixed precision** training with native PyTorch AMP
- **Modern augmentations** (RandAugment, AutoAugment, CutMix, Mixup)
- **Advanced regularization** (DropPath, DropBlock, Label Smoothing)
- **Flexible learning rate scheduling**
- **Model EMA** (Exponential Moving Average)
- **Gradient accumulation**
- **Checkpointing and resuming**

## Training Script

### Basic Usage

```bash
python train.py /path/to/imagenet \
  --model resnet50 \
  --batch-size 128 \
  --lr 0.1 \
  --epochs 100 \
  --amp
```

### Common Arguments

#### Model Configuration

```bash
--model resnet50              # Model architecture
--pretrained                  # Start from pretrained weights
--num-classes 1000           # Number of output classes
--img-size 224               # Input image size
--in-chans 3                 # Number of input channels
```

#### Data Configuration

```bash
--data-dir /path/to/data     # Dataset directory
--dataset ImageFolder        # Dataset type
--train-split train          # Training split name
--val-split val              # Validation split name
--batch-size 128             # Batch size per GPU
--workers 4                  # Data loading workers
```

#### Optimizer Settings

```bash
--opt adamw                  # Optimizer (adamw, sgd, lamb, etc.)
--lr 0.001                   # Base learning rate
--weight-decay 0.05          # Weight decay
--momentum 0.9               # Momentum (for SGD)
--clip-grad 1.0              # Gradient clipping threshold
--clip-mode norm             # Clipping mode (norm, value, agc)
```

#### Learning Rate Schedule

```bash
--sched cosine               # LR scheduler (cosine, step, plateau)
--epochs 300                 # Number of training epochs
--warmup-epochs 5            # Warmup epochs
--warmup-lr 1e-6            # Warmup learning rate
--min-lr 1e-6               # Minimum learning rate
--decay-rate 0.1            # Decay rate for step scheduler
```

#### Augmentation

```bash
--aa rand-m9-mstd0.5        # AutoAugment policy
--reprob 0.25               # Random erasing probability
--remode pixel              # Random erasing mode
--mixup 0.8                 # Mixup alpha
--cutmix 1.0                # CutMix alpha
--mixup-prob 1.0            # Probability of mixup/cutmix
--smoothing 0.1             # Label smoothing
```

#### Regularization

```bash
--drop 0.0                  # Dropout rate
--drop-path 0.1             # DropPath rate
--drop-block 0.0            # DropBlock rate
```

#### Training Features

```bash
--amp                       # Use mixed precision training
--channels-last             # Use channels last memory format
--model-ema                 # Track EMA of model weights
--model-ema-decay 0.9998   # EMA decay rate
```

#### Checkpointing

```bash
--output /path/to/output    # Output directory
--checkpoint-hist 10        # Keep last N checkpoints
--resume /path/to/ckpt     # Resume from checkpoint
```

### Advanced Examples

#### Training ResNet-50 with Modern Recipe

```bash
python train.py /imagenet \
  --model resnet50 \
  --batch-size 128 \
  --opt adamw \
  --lr 0.001 \
  --weight-decay 0.05 \
  --sched cosine \
  --epochs 300 \
  --warmup-epochs 5 \
  --aa rand-m9-mstd0.5 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --drop-path 0.1 \
  --amp \
  --model-ema \
  --workers 8
```

#### Training Vision Transformer

```bash
python train.py /imagenet \
  --model vit_base_patch16_224 \
  --batch-size 256 \
  --opt adamw \
  --lr 0.001 \
  --weight-decay 0.05 \
  --sched cosine \
  --epochs 300 \
  --warmup-epochs 5 \
  --warmup-lr 1e-6 \
  --min-lr 1e-5 \
  --aa rand-m9-mstd0.5-inc1 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --drop-path 0.1 \
  --amp \
  --model-ema \
  --model-ema-decay 0.99996
```

#### Fine-tuning on Custom Dataset

```bash
python train.py /path/to/custom/dataset \
  --model resnet50 \
  --pretrained \
  --num-classes 10 \
  --batch-size 64 \
  --opt adamw \
  --lr 0.0001 \
  --weight-decay 0.01 \
  --epochs 50 \
  --warmup-epochs 3 \
  --aa rand-m7-mstd0.5 \
  --mixup 0.2 \
  --cutmix 0.0 \
  --smoothing 0.1 \
  --amp
```

## Validation Script

### Basic Usage

```bash
python validate.py /path/to/imagenet \
  --model resnet50 \
  --checkpoint /path/to/checkpoint.pth \
  --batch-size 256 \
  --workers 4
```

### Common Arguments

```bash
--model resnet50             # Model architecture
--checkpoint path.pth        # Path to checkpoint
--pretrained                 # Use pretrained weights
--num-classes 1000          # Number of classes
--batch-size 256            # Validation batch size
--workers 4                 # Number of workers
--img-size 224              # Input image size
--crop-pct 0.875            # Center crop percentage
--interpolation bicubic     # Resize interpolation
--amp                       # Use mixed precision
--channels-last             # Use channels last format
--results-file results.csv  # Save results to CSV
```

### Test-Time Augmentation

```bash
python validate.py /imagenet \
  --model resnet50 \
  --checkpoint checkpoint.pth \
  --batch-size 64 \
  --img-size 288 \  # Larger than training size
  --crop-pct 1.0 \  # Use full image
  --amp
```

## Inference Script

### Basic Usage

```bash
python inference.py /path/to/images \
  --model resnet50 \
  --checkpoint checkpoint.pth \
  --output predictions.csv
```

### Batch Inference

```bash
python inference.py /path/to/images \
  --model resnet50 \
  --pretrained \
  --batch-size 32 \
  --output predictions.csv \
  --topk 5  # Save top-5 predictions
```

## Distributed Training

### Single Node, Multiple GPUs

```bash
./distributed_train.sh 4 /imagenet \
  --model resnet50 \
  --batch-size 64 \
  --lr 0.1 \
  --epochs 100
```

The script automatically distributes across 4 GPUs (specified by the first argument).

### Multi-Node Training

```bash
# Node 0
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="192.168.1.1" \
  --master_port=12345 \
  train.py /imagenet --model resnet50 ...

# Node 1
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr="192.168.1.1" \
  --master_port=12345 \
  train.py /imagenet --model resnet50 ...
```

## Training Tips

### Learning Rate Scaling

When using larger batch sizes, scale the learning rate:

```bash
# Base: batch_size=128, lr=0.1
# Scaled: batch_size=512, lr=0.4 (4x batch = 4x lr)
python train.py /imagenet \
  --batch-size 512 \
  --lr 0.4
```

### Gradient Accumulation

For large models with limited memory:

```bash
python train.py /imagenet \
  --model vit_large_patch16_224 \
  --batch-size 32 \  # Effective batch = 32 * 4 = 128
  --grad-accum-steps 4 \
  --amp
```

### Layer-wise Learning Rates

Different learning rates for different layers:

```bash
python train.py /imagenet \
  --model vit_base_patch16_224 \
  --lr 0.001 \
  --layer-decay 0.75  # Earlier layers use lower LR
```

### Model EMA Best Practices

```bash
python train.py /imagenet \
  --model-ema \
  --model-ema-decay 0.9998 \  # For batch_size >= 128
  --model-ema-force-cpu  # Keep EMA on CPU to save GPU memory
```

## Custom Datasets

### Directory Structure

```
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── class1/
    └── class2/
```

### Training on Custom Data

```bash
python train.py /path/to/dataset \
  --dataset ImageFolder \
  --train-split train \
  --val-split val \
  --num-classes NUM_CLASSES \
  --model resnet50 \
  --pretrained
```

## Monitoring Training

### TensorBoard

```bash
python train.py /imagenet \
  --log-interval 50 \
  --output /path/to/output

# In another terminal
tensorboard --logdir /path/to/output
```

### Weights & Biases

```bash
python train.py /imagenet \
  --log-wandb \
  --wandb-project my-project
```

## Checkpointing

### Save Checkpoints

Checkpoints are automatically saved to the output directory:

```
output/
├── checkpoint-0.pth.tar
├── checkpoint-10.pth.tar
├── checkpoint-20.pth.tar
└── model_best.pth.tar
```

### Resume Training

```bash
python train.py /imagenet \
  --resume /path/to/checkpoint.pth.tar
```

### Start from Checkpoint with Different Settings

```bash
python train.py /imagenet \
  --initial-checkpoint /path/to/checkpoint.pth.tar \
  --lr 0.0001  # New learning rate
```

## Exporting Models

### Export to ONNX

```python
import torch
import timm

model = timm.create_model('resnet50', pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### Export to TorchScript

```python
model = timm.create_model('resnet50', pretrained=True)
model.eval()

# Trace
example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example)
traced.save('resnet50_traced.pt')

# Script (for dynamic control flow)
scripted = torch.jit.script(model)
scripted.save('resnet50_scripted.pt')
```

## Performance Optimization

### Mixed Precision Training

Significantly faster on modern GPUs:

```bash
--amp  # Enable automatic mixed precision
```

### Channels Last Memory Format

Better performance on GPUs with Tensor Cores:

```bash
--channels-last
```

### Fused Optimizers

Use APEX fused optimizers for better performance:

```bash
--opt fusedadam  # Requires NVIDIA APEX
```

### Compile with TorchDynamo (PyTorch 2.0+)

```bash
--torchcompile  # Enable torch.compile()
```

## Links

- [Getting Started](getting-started.md)
- [Model Features](features.md)
- [Model Metrics](metrics/overview.md)
- [Official Training Documentation](https://huggingface.co/docs/timm/training_script)
