import tonic
import torch
import numpy as np
from collections import defaultdict
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
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=20),
        ]
    )
    trainset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=True
    )
    testset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=False
    )
    trainset = MemoryCachedDataset(
        trainset,
        transform=tonic.transforms.Compose(
            [torch.from_numpy, RandomRotation([-10, 10])]
        ),
    )
    testset = MemoryCachedDataset(testset)
    return trainset, testset


def non_iid_partition(trainSet, num_clients, num_classes=10, samples_per_class=1000):
    targets = np.array(trainSet.targets)
    class_indices = {i: np.where(targets == i)[0] for i in range(num_classes)}

    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        indices = class_indices[cls]
        np.random.shuffle(indices)
        split = np.array_split(indices, num_clients)
        for client_id in range(num_clients):
            client_indices[client_id].extend(split[client_id])

    client_datasets = [torch.utils.data.Subset(trainSet, inds) for inds in client_indices]
    return client_datasets

def dirichlet_non_iid_partition(trainSet, num_clients, num_classes=10, alpha=0.5):
    targets = np.array(trainSet.targets)
    class_indices = {i: np.where(targets == i)[0] for i in range(num_classes)}

    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        indices = class_indices[cls]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(indices)).astype(int)

        # Fix any rounding issues
        while proportions.sum() > len(indices):
            proportions[np.argmax(proportions)] -= 1
        while proportions.sum() < len(indices):
            proportions[np.argmin(proportions)] += 1

        split = np.split(indices, np.cumsum(proportions)[:-1])
        for client_id, client_split in enumerate(split):
            client_indices[client_id].extend(client_split)

    client_datasets = [torch.utils.data.Subset(trainSet, inds) for inds in client_indices]
    return client_datasets

def load_dataset(
    name, path, num_partitions: int, batch_size: int, val_ratio: float = 0.1, non_iid = True
):
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
    if non_iid:
        trainsets = dirichlet_non_iid_partition(trainSet, num_partitions)
    else:
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
                num_workers=2,
                collate_fn=collate_fn,
            )
        )
        validationList.append(
            DataLoader(
                for_validation,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn,
            )
        )

    print("BATCH SIZE: ", batch_size)
    testList = DataLoader(
        testSet,
        batch_size=batch_size,  # originally 128
        collate_fn=collate_fn,
    )
    return trainingList, validationList, testList
