from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop, Pad, Normalize, ToTensor


def get_train_test_loaders(dataset_name, path, batch_size, num_workers):

    assert dataset_name in datasets.__dict__, "Unknown dataset name {}".format(dataset_name)
    fn = datasets.__dict__[dataset_name]

    train_transform = Compose([
        Pad(2),
        RandomCrop(32),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = fn(root=path, train=True, transform=train_transform, download=True)
    test_ds = fn(root=path, train=False, transform=test_transform, download=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
