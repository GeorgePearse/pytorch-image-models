# Features

`timm` provides a comprehensive set of features for computer vision research and production.

## Core Features

### 1000+ Pre-trained Models

Access to a vast collection of state-of-the-art models:

```python
import timm

# List all available models
all_models = timm.list_models()
print(f"Total models: {len(all_models)}")

# Search for specific models
resnet_models = timm.list_models('resnet*')
vit_models = timm.list_models('vit_*')
```

### Consistent Model API

All models share a common interface:

```python
# Create any model with pretrained weights
model = timm.create_model('resnet50', pretrained=True)

# Access/change classifier
num_classes = model.get_classifier()
model.reset_classifier(num_classes=10)

# Feature extraction
features = model.forward_features(x)
```

### Multi-Scale Feature Extraction

Easy feature pyramid extraction from any model:

```python
# Create model for feature extraction
model = timm.create_model(
    'resnet50',
    features_only=True,
    out_indices=(1, 2, 3, 4)
)

# Get multi-scale features
features = model(x)
for feat in features:
    print(feat.shape)  # Different spatial resolutions
```

### Flexible Image Preprocessing

Built-in data configuration for each model:

```python
# Get model-specific preprocessing config
data_config = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**data_config)

# Apply preprocessing
preprocessed = transform(image)
```

## Model Features

### Adaptive Weight Loading

Automatically adapts weights to different configurations:

```python
# Load pretrained weights with different classifier
model = timm.create_model(
    'resnet50',
    pretrained=True,
    num_classes=100  # Adapts final layer
)

# Load with different input channels
model = timm.create_model(
    'resnet50',
    pretrained=True,
    in_chans=1  # Converts RGB weights to grayscale
)
```

### Dynamic Resolution

Many models support different input sizes:

```python
# Standard resolution
model = timm.create_model('vit_base_patch16_224', img_size=224)

# Higher resolution
model = timm.create_model('vit_base_patch16_224', img_size=384)
```

### Test Time Augmentation

Wrapper for improved inference with larger images:

```python
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Create model with test-time pooling
model = timm.create_model('resnet50', pretrained=True)
config = resolve_data_config({}, model=model)

# Use larger input size at test time
config['input_size'] = (3, 288, 288)
transform = create_transform(**config)
```

## Optimizers

Extensive selection of modern optimizers via `timm.optim.create_optimizer_v2`:

```python
import timm.optim

# List all available optimizers
optimizers = timm.optim.list_optimizers()

# Create optimizer
optimizer = timm.optim.create_optimizer_v2(
    model,
    opt='adamw',
    lr=1e-3,
    weight_decay=0.05
)
```

### Available Optimizers

- **AdamW**: Adam with decoupled weight decay
- **LAMB**: Layer-wise Adaptive Moments optimizer
- **LARS**: Layer-wise Adaptive Rate Scaling
- **Lion**: Evolved optimizer from Google
- **Adafactor**: Memory-efficient adaptive optimizer
- **AdaBelief**: Adapting stepsizes by belief in gradients
- **Adan**: Adaptive Nesterov momentum
- **ADOPT**: Adaptive gradient methods
- **MARS**: Modern adaptive optimizer
- **LaProp**: Layer-wise adaptive learning rates
- **Muon**: Orthogonalization-based optimizer
- **Kron**: Kronecker-factored preconditioner

## Augmentations

State-of-the-art data augmentation techniques:

### Random Erasing

```python
from timm.data import RandomErasing

transform = RandomErasing(
    probability=0.25,
    mode='pixel',
    max_count=1
)
```

### Mixup & CutMix

```python
from timm.data import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    label_smoothing=0.1,
    num_classes=1000
)
```

### AutoAugment & RandAugment

```python
from timm.data.auto_augment import rand_augment_transform

transform = rand_augment_transform(
    config_str='rand-m9-mstd0.5',
    hparams={'translate_const': 100}
)
```

## Regularization

### DropPath (Stochastic Depth)

```python
from timm.models.layers import DropPath

drop_path = DropPath(drop_prob=0.1)
```

### DropBlock

```python
from timm.models.layers import DropBlock2d

drop_block = DropBlock2d(
    drop_prob=0.1,
    block_size=7
)
```

### Blur Pooling

Anti-aliased downsampling for better shift-invariance:

```python
# Many models support anti-aliased pooling
model = timm.create_model(
    'resnet50',
    aa_layer=True  # Enable blur pooling
)
```

## Learning Rate Schedulers

Flexible scheduling with warmup and restarts:

```python
from timm.scheduler import CosineLRScheduler

scheduler = CosineLRScheduler(
    optimizer,
    t_initial=300,  # epochs
    lr_min=1e-6,
    warmup_t=5,
    warmup_lr_init=1e-6,
    cycle_limit=1
)
```

### Available Schedulers

- **Cosine**: Cosine annealing with warm restarts
- **Step**: Step decay at milestones
- **Plateau**: Reduce on plateau
- **Tanh**: Tanh decay with restarts
- **Polynomial**: Polynomial decay

## Training Utilities

### Model EMA

Exponential moving average of model weights:

```python
from timm.utils import ModelEmaV2

model_ema = ModelEmaV2(
    model,
    decay=0.9999,
    device=device
)

# Update during training
model_ema.update(model)

# Use for validation
with torch.no_grad():
    output = model_ema.module(input)
```

### Gradient Clipping

Adaptive gradient clipping from NFNets:

```python
from timm.utils import dispatch_clip_grad

dispatch_clip_grad(
    model.parameters(),
    value=1.0,
    mode='agc'  # Adaptive Gradient Clipping
)
```

### Mixed Precision Training

Native PyTorch AMP support:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in loader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Attention Modules

Extensive selection of attention mechanisms:

- **Squeeze-and-Excitation (SE)**: Channel attention
- **Effective SE (ESE)**: Efficient variant
- **CBAM**: Convolutional Block Attention Module
- **ECA**: Efficient Channel Attention
- **Global Context (GC)**: Global context modeling
- **Gather-Excite (GE)**: Spatial-channel attention
- **Selective Kernel (SK)**: Dynamic convolution selection
- **SPLAT**: Split attention
- **Coordinate Attention**: Position-aware attention

## Layer Types

Rich collection of layer implementations:

### Convolution Variants

- Standard Conv2d with same padding
- Depthwise separable convolutions
- Mixed depthwise convolutions
- Blur pooling
- Space-to-depth transforms

### Normalization

- BatchNorm with sync support
- GroupNorm
- LayerNorm (2D)
- RMSNorm
- EvoNorm

### Activation Functions

- All PyTorch activations
- Swish / SiLU
- Mish
- Hard Swish
- Hard Sigmoid
- GELU variants

## Utilities

### Model Surgery

```python
# Freeze layers
for name, param in model.named_parameters():
    if 'layer4' not in name:
        param.requires_grad = False

# Replace layers
model.fc = nn.Linear(2048, num_classes)
```

### Feature Extraction

```python
# Extract intermediate features
model = timm.create_model('resnet50', features_only=True)
features = model(x)

# Or use forward_intermediates
model = timm.create_model('vit_base_patch16_224')
final_feat, intermediates = model.forward_intermediates(x)
```

### Model Information

```python
# Get model metadata
model_info = timm.models.get_pretrained_cfg('resnet50')

# Count parameters
num_params = sum(p.numel() for p in model.parameters())

# Get feature info
if hasattr(model, 'feature_info'):
    for info in model.feature_info:
        print(f"Stage: {info['module']}, Channels: {info['num_chs']}")
```

## Production Features

### ONNX Export

```python
# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)
```

### TorchScript

```python
# Script model
scripted = torch.jit.script(model)

# Trace model
traced = torch.jit.trace(model, example_input)
```

### Quantization Support

Many models support PyTorch quantization for inference speedup.

## Documentation & Resources

### Model Hub

All pretrained weights are available on Hugging Face Hub:

```python
# Models are automatically downloaded from HF Hub
model = timm.create_model('resnet50.a1_in1k', pretrained=True)

# Or explicitly use hub:
model = timm.create_model('hf-hub:timm/resnet50.a1_in1k', pretrained=True)
```

### Papers with Code

Browse models by task and performance: https://paperswithcode.com/lib/timm

### Official Documentation

Comprehensive guides at: https://huggingface.co/docs/timm

## Links

- [Model Architectures](architectures.md)
- [Model Metrics](metrics/overview.md)
- [Getting Started](getting-started.md)
- [Training Scripts](training.md)
