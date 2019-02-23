# MovingMNIST

Simple PyTorch dataset of [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) dataset.
With auto download.

## Example

```
import torch
from MovingMNIST import MovingMNIST

train_set = MovingMNIST(root='.data/mnist', train=True, download=True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=100,
                 shuffle=True)

```
