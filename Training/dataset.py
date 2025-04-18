import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Normalize, ToTensor


def get_MNISTdataset(path):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(root=path, train=True, download=True, transform=tr)
    test = datasets.MNIST(root=path, train=False, download=True, transform=tr)
    return train, test


def get_CIFARdataset(path):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = datasets.CIFAR10(root=path, train=True, download=True, transform=tr)
    test = datasets.CIFAR10(root=path, train=False, download=True, transform=tr)
    return train, test


def load_dataset(
    name, path, num_partitions: int, batch_size: int, val_ratio: float = 0.1
):
    """val_ratio is used to vary data amount among clients"""
    transform = transforms.ToTensor()

    if name == "mnist":
        trainSet, testSet = get_MNISTdataset(path)
    elif name == "cifar10":
        trainSet, testSet = get_CIFARdataset(path)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # split trainset into partitioned trainsets - below has equal size partitions but can be changed
    num_images = len(trainSet) // num_partitions
    partition_length = [num_images] * num_partitions
    trainsets = random_split(
        trainSet, partition_length, torch.Generator().manual_seed(42)
    )  # For repetition

    # partition into train and validation sets
    trainingList = []
    validationList = []
    for partition in trainsets:
        total = len(partition)
        validation_length = int(val_ratio * total)
        train_length = total - validation_length
        for_training, for_validation = random_split(
            partition,
            [train_length, validation_length],
            generator=torch.Generator().manual_seed(42),
        )
        trainingList.append(
            DataLoader(for_training, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        validationList.append(
            DataLoader(
                for_validation, batch_size=batch_size, shuffle=False, num_workers=2
            )
        )

    testList = DataLoader(testSet, batch_size=128)
    return trainingList, validationList, testList
