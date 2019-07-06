from torchvision import models

from cifar.dataloaders import get_train_test_loaders as get_cifar_train_test_loaders
from cifar import fastresnet


def set_seed(seed):

    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_train_test_loaders(dataset_name, path, batch_size, num_workers):

    if "cifar" in dataset_name.lower():
        return get_cifar_train_test_loaders(dataset_name, path, batch_size, num_workers)

    raise RuntimeError("Unknown dataset '{}'".format(dataset_name))


def get_model(name):
    
    fn = None
    if name in models.__dict__:
        fn = models.__dict__[name]
    elif name in fastresnet.__dict__:
        fn = fastresnet.__dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn()
