## LeNet5 Architecture

### Body
This implements the convolutional part of the network.
Which consists in 2 convolutional layers. Each convolutional layer is followed by a pooling layer. It also works as a feature extractor, in classical Machine Learning terms.

### Head
This implements the fully connected part of the network. Consist of 3 fully connected layers, with the last layer having the 10 output classes. It also works as a classifier.

## Convolutional Layer function synthax
````
torch.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size, 
    stride=1, 
    padding=0, 
    dilation=1, 
    groups=1, 
    bias=True, 
    padding_mode='zeros'
)
````
- in_channels: number of input channels, for colored image input channels must be 3.
- out_channels: number of filters
- kernel_size: ``int`` or ``tuple of int``. As in `kernel_size`, if it is `int` same padding will be done accross the height and width
---
## Annotated Overview (auto-appended)
_Generated on 2025-09-21 03:37:27_

This section explains the purpose of each file. Only comments/docstrings were added to the code; no logic or behavior was changed.

- **lenet5.py**: Defines a LeNet‑5 style CNN with convolution + pooling body and fully connected head for MNIST‑like classification.
- **main.py**: Entrypoint wiring: sets up system/configs, data loaders, model, and kicks off training/inference.
- **system_configuration.py**: Global system toggles like random seeds and cuDNN determinism/benchmark flags.
- **trainer.py**: Training/validation loop utilities (model train/eval modes, loss/accuracy tracking, device movement).
- **training_configuration.py**: Hyperparameters and runtime settings (batch size, epochs, num_workers, device).
