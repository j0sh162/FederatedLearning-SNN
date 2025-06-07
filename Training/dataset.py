from collections import defaultdict

import numpy as np
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
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=20),
        ]
    )

    train_dataset = tonic.datasets.NMNIST(
        save_to="./data", train=True, transform=frame_transform
    )

    testset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=False
    )
    # trainset = MemoryCachedDataset(
    #     trainset,
    #     transform=tonic.transforms.Compose(
    #         [frame_transform, torch.from_numpy, RandomRotation([-10, 10])]
    #     ),
    # )
    # testset = MemoryCachedDataset(testset)
    return train_dataset, testset


"""def get_NMNIST_dataset(path, limit=1000):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = tonic.transforms.Compose(
        [
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )
    trainset_full = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=True
    )
    testset_full = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=False
    )

    # Restrict both to a smaller subset for quick testing
    trainset, _ = random_split(
        trainset_full,
        [limit, len(trainset_full) - limit],
        generator=torch.Generator().manual_seed(42),
    )
    testset, _ = random_split(
        testset_full,
        [limit, len(testset_full) - limit],
        generator=torch.Generator().manual_seed(42),
    )

    trainset = MemoryCachedDataset(
        trainset,
        transform=tonic.transforms.Compose(
            [torch.from_numpy, RandomRotation([-10, 10])]
        ),
    )
    testset = MemoryCachedDataset(testset)

    return trainset, testset"""


def get_dataset_targets(dataset):
    """Retrieve labels from dataset even if it doesn't have a 'targets' attribute."""
    try:
        return dataset.targets  # Works for standard torch datasets
    except AttributeError:
        # Fallback: collect labels by indexing
        return [dataset[i][1] for i in range(len(dataset))]


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

    client_datasets = [
        torch.utils.data.Subset(trainSet, inds) for inds in client_indices
    ]
    return client_datasets


def dirichlet_non_iid_partition(
    dataset, num_clients, num_classes=10, alpha=0.5, min_size=1
):
    targets = np.array(get_dataset_targets(dataset))
    class_indices = [np.where(targets == y)[0] for y in range(num_classes)]

    while True:
        client_idx = [[] for _ in range(num_clients)]

        for idx_y in class_indices:
            if len(idx_y) == 0:
                continue
            np.random.shuffle(idx_y)
            counts = np.random.multinomial(
                len(idx_y), np.random.dirichlet([alpha] * num_clients)
            )
            start = 0
            for cid, cnt in enumerate(counts):
                client_idx[cid].extend(idx_y[start : start + cnt])
                start += cnt

        sizes = [len(idxs) for idxs in client_idx]
        if min(sizes) >= min_size:
            break  # ensure no partition is empty

    return [torch.utils.data.Subset(dataset, ids) for ids in client_idx]


def load_dataset(
    name,
    path,
    num_partitions: int,
    batch_size: int,
    val_ratio: float = 0.1,
    non_iid=True,
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
