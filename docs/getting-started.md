# Getting Started

This guide will help you get started with PyTorch Image Models (timm).

## Installation

### Basic Installation

Install from PyPI:

```bash
pip install timm
```

### Development Installation

Clone and install from source:

```bash
git clone https://github.com/GeorgePearse/pytorch-image-models
cd pytorch-image-models
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- torchvision

## Quick Start

### Loading a Pretrained Model

```python
import timm
import torch

# Create a model with pretrained weights
model = timm.create_model('resnet50', pretrained=True)
model.eval()

# Prepare input (batch_size=1, channels=3, height=224, width=224)
x = torch.randn(1, 3, 224, 224)

# Get predictions
with torch.no_grad():
    output = model(x)

print(output.shape)  # torch.Size([1, 1000])
```

### Listing Available Models

```python
import timm

# List all models
all_models = timm.list_models()
print(f"Total models: {len(all_models)}")

# Search for specific models
resnet_models = timm.list_models('resnet*')
print(f"ResNet models: {len(resnet_models)}")

# List models with pretrained weights
pretrained_models = timm.list_models(pretrained=True)

# Filter by specific characteristics
vit_models = timm.list_models('vit_*', pretrained=True)
```

### Loading with Custom Number of Classes

```python
# For fine-tuning on custom dataset
model = timm.create_model(
    'resnet50',
    pretrained=True,
    num_classes=10  # Your number of classes
)
```

## Image Preprocessing

### Using Model-Specific Transforms

```python
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

# Create model
model = timm.create_model('resnet50', pretrained=True)
model.eval()

# Get model-specific preprocessing config
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic',
#  'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), ...}

# Create transform
transform = create_transform(**config)

# Load and preprocess image
img = Image.open('path/to/image.jpg')
tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    output = model(tensor)
```

### Manual Preprocessing

```python
from torchvision import transforms

# Standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Feature Extraction

### Extract Features Only

```python
# Create model for feature extraction
model = timm.create_model(
    'resnet50',
    pretrained=True,
    num_classes=0,  # Remove classifier
    global_pool=''   # Remove global pooling
)

# Get features
features = model.forward_features(x)
print(features.shape)  # torch.Size([1, 2048, 7, 7])
```

### Multi-Scale Feature Extraction

```python
# Create feature pyramid model
model = timm.create_model(
    'resnet50',
    features_only=True,
    out_indices=(1, 2, 3, 4)  # Get features from these stages
)

# Get multi-scale features
features = model(x)

for i, feat in enumerate(features):
    print(f"Feature {i}: {feat.shape}")
# Feature 0: torch.Size([1, 256, 56, 56])
# Feature 1: torch.Size([1, 512, 28, 28])
# Feature 2: torch.Size([1, 1024, 14, 14])
# Feature 3: torch.Size([1, 2048, 7, 7])
```

### Using forward_intermediates (ViT models)

```python
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Get final features and all intermediate layer outputs
final_feat, intermediates = model.forward_intermediates(x)

print(final_feat.shape)  # torch.Size([1, 197, 768])

for i, feat in enumerate(intermediates):
    print(f"Layer {i}: {feat.shape}")
```

## Fine-Tuning

### Basic Fine-Tuning Setup

```python
import torch
import torch.nn as nn
import timm

# Load pretrained model
model = timm.create_model(
    'resnet50',
    pretrained=True,
    num_classes=10  # Your dataset classes
)

# Setup for fine-tuning
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### Freeze Early Layers

```python
# Freeze all layers except final classifier
for name, param in model.named_parameters():
    if 'fc' not in name:  # 'fc' is the classifier layer name
        param.requires_grad = False

# Only optimize classifier
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

### Progressive Unfreezing

```python
# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier
for param in model.get_classifier().parameters():
    param.requires_grad = True

# Train classifier first...

# Later, unfreeze more layers
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
```

## Using Different Input Sizes

### Runtime Resolution Change

```python
# Create model with default resolution
model = timm.create_model('resnet50', pretrained=True)

# Use with different input size (must be divisible by 32 for ResNet)
x_small = torch.randn(1, 3, 224, 224)
x_large = torch.randn(1, 3, 448, 448)

output_small = model(x_small)
output_large = model(x_large)
```

### Set Input Size at Creation

```python
# For models that need specific input size (like ViT)
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    img_size=384  # Use 384x384 instead of 224x224
)
```

## Working with Different Input Channels

### Single Channel Input

```python
# For grayscale images
model = timm.create_model(
    'resnet50',
    pretrained=True,
    in_chans=1  # Single channel
)
```

### Multi-Channel Input

```python
# For multi-spectral images
model = timm.create_model(
    'resnet50',
    pretrained=True,
    in_chans=4  # 4 channels
)
```

## GPU Usage

### Single GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Inference
x = x.to(device)
output = model(x)
```

### Multi-GPU (DataParallel)

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to('cuda')
```

### Distributed Training (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Create model and move to GPU
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

## Model Surgery

### Replace Classifier

```python
import torch.nn as nn

# Get classifier
classifier = model.get_classifier()
print(classifier)

# Replace with custom head
model.reset_classifier(num_classes=100)

# Or manually replace
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 100)
)
```

### Add Auxiliary Heads

```python
class ModelWithAuxHead(nn.Module):
    def __init__(self, base_model, num_classes=10):
        super().__init__()
        self.base = base_model
        self.aux_head = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.base.forward_features(x)
        # Main output
        main_out = self.base.forward_head(features)
        # Auxiliary output from intermediate features
        aux_out = self.aux_head(features.mean(dim=(2, 3)))
        return main_out, aux_out
```

## Inference Tips

### Batch Inference

```python
import torch.nn.functional as F

model.eval()
all_predictions = []

with torch.no_grad():
    for batch in data_loader:
        images = batch.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
```

### Mixed Precision Inference

```python
from torch.cuda.amp import autocast

model.eval()
with torch.no_grad(), autocast():
    output = model(x.cuda())
```

### Test Time Augmentation

```python
import torch.nn.functional as F

def tta_inference(model, image, num_augments=5):
    """Test time augmentation with random crops and flips"""
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original
        pred = model(image)
        predictions.append(F.softmax(pred, dim=1))

        # Horizontal flip
        pred = model(torch.flip(image, dims=[3]))
        predictions.append(F.softmax(pred, dim=1))

        # Random crops (example)
        for _ in range(num_augments - 2):
            # Apply random transformations
            pred = model(image)
            predictions.append(F.softmax(pred, dim=1))

    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred
```

## Common Issues

### Out of Memory

```python
# Reduce batch size
batch_size = 16  # Try smaller values

# Use gradient checkpointing (for supported models)
model.set_grad_checkpointing(enable=True)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(x)
```

### Slow Training

```python
# Use DataLoader with multiple workers
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Adjust based on CPU cores
    pin_memory=True
)

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True
```

## Next Steps

- [Explore Model Architectures](architectures.md)
- [Check Model Performance](metrics/overview.md)
- [Learn About Features](features.md)
- [Read Training Scripts Guide](training.md)

## Additional Resources

- [Official timm Documentation](https://huggingface.co/docs/timm)
- [Papers with Code](https://paperswithcode.com/lib/timm)
- [Hugging Face Model Hub](https://huggingface.co/timm)
