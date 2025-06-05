import tonic
import torch
from tonic import MemoryCachedDataset
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Normalize, RandomRotation, ToTensor


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


def get_NMNIST_dataset(path):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = tonic.transforms.Compose(
        [
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )
    trainset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=True
    )
    testset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=False
    )
    split = 0.5
    trainset = random_split(
        trainset,
        [split, 1 - split],
        generator=torch.Generator().manual_seed(42),
    )[0]
    testset = random_split(
        testset,
        [split, 1 - split],
        generator=torch.Generator().manual_seed(42),
    )[0]
    trainset = MemoryCachedDataset(
        trainset,
        transform=tonic.transforms.Compose(
            [torch.from_numpy, RandomRotation([-10, 10])]
        ),
    )
    testset = MemoryCachedDataset(testset)
    return trainset, testset


def load_dataset(name, path, num_partitions: int, batch_size: int, test_batch_size: int, val_ratio: float = 0.1):
    """val_ratio is used to vary data amount among clients"""
    transform = transforms.ToTensor()

    if name == "mnist":
        trainSet, testSet = get_MNISTdataset(path)
        collate_fn = None
    elif name == "cifar10":
        trainSet, testSet = get_CIFARdataset(path)
        collate_fn = None
    elif name == "NMNIST":
        trainSet, testSet = get_NMNIST_dataset(path)
        collate_fn = tonic.collation.PadTensors(batch_first=False)
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
            DataLoader(
                for_training,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
            )
        )
        validationList.append(
            DataLoader(
                for_validation,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
        )

    print("BATCH SIZE: ", batch_size)
    testList = DataLoader(
        testSet,
        batch_size=test_batch_size,  # originally 128
        collate_fn=collate_fn,
    )
    return trainingList, validationList, testList
