# Research purpose implementation of Tensor Train and Tensor Ring algorithms in Pytorch

[Paper](https://arxiv.org/pdf/1901.10787.pdf)

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies and Getting Started
- ``torch >= 1.0.0``
- ``torchvision >= 0.2.1``
- ``numpy``
- ``sympy``
- ``scipy``

You may install `PyTorch` using any suggested method for your environment [here](https://pytorch.org/get-started/locally/).

Also, after cloning the repo, you can run  ``setup.py install`` in the command line to install the required packages.

### Setting up experiments

To check the experiments settings, see a file ``experiments.sh``.

For example, to run an experiment for TT embedding layer you can run:

```sh
python train.py --embed_dim 256 --dataset imdb --embedding tt \
    --n_epochs 100 --d 3 --ranks 16 --gpu 1
```

### Repository structure

The directory `t3nsor` contains classes and function for TT and TR decompositions, embedding layers and so on.
The directory `sentiment` contains the models and experiment setting files.

## Authors
- Valentin Khrulkov
- Oleksii Hrinchuk
- Leyla Mirvakhabova
- Elena Orlova
- Ivan Oseledets
