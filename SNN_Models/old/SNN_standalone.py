# Based on: https://ieeexplore.ieee.org/abstract/document/10242251
# Adapted from: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tonic
import tonic.transforms as transforms
import torch
import torchvision
from rich import print
from snntorch import functional as SF
from snntorch import surrogate
from tonic import MemoryCachedDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from SNN_Models import SNN
from SNN_Models.SNN_utils import test
from Training import dataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_iters", type=int, default=-1)
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

writer = SummaryWriter(f"runs/SurrogateGradient_centralized/{args.seed}")

sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
trainset, testset = dataset.get_NMNIST_dataset("./data")

transform = tonic.transforms.Compose(
    [
        # torch.from_numpy,
        # torchvision.transforms.RandomRotation([-10, 10]),
    ]
)

cached_trainset = MemoryCachedDataset(trainset, transform=transform)

# no augmentations for the testset
cached_testset = MemoryCachedDataset(testset)

batch_size = args.batchsize
train_loader = DataLoader(
    cached_trainset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=False,
)
test_loader = DataLoader(
    cached_testset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
)

# train_loader, test_loader = dataset.load_dataset()

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

snn_net = SNN.Net()

if False:  # os.path.exists("snn_net.pt"):
    print("Loading existing model weights")
    # Load the model weights
    snn_net.net.load_state_dict(
        torch.load("snn_net.pt", weights_only=True, map_location=device)
    )

optimizer = torch.optim.AdamW(snn_net.net.parameters(), lr=0.0002, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = args.epochs
num_iters = args.num_iters
num_batches = len(train_loader)

hist = []

# training loop
for epoch in range(num_epochs):
    for batch_number, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        snn_net.net.train()
        spk_rec = snn_net.forward(data)
        loss_val = loss_fn(spk_rec, targets)
        acc = SF.accuracy_rate(spk_rec, targets)

        writer.add_scalar(
            "Loss/train",
            loss_val.item(),
            epoch * num_batches + batch_number,
        )
        writer.add_scalar(
            "Acc/train",
            acc,
            epoch * num_batches + batch_number,
        )
        hist.append([epoch, batch_number, loss_val.item(), acc, "train"])

        print(
            "Epoch {:02d} | Batch {:03d}/{:03d} | Loss {:05.2f} | Accuracy {:05.2f}%".format(
                epoch, batch_number, num_batches, loss_val.item(), acc * 100
            )
        )

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # training loop breaks after num_iters iterations/ batches
        if batch_number == num_iters - 1:
            break

test_loss, test_acc = test(snn_net, test_loader, device)
hist.append([epoch, batch_number, test_loss, test_acc, "test"])
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")


writer.flush()
writer.close()
torch.save(snn_net.net.state_dict(), "snn_net.pt")

# fig = plt.figure(facecolor="w")
# df = pd.DataFrame(hist, columns=["epoch", "batch", "Loss", "Acc", "Set"])
# df.to_csv("acc_hist.csv", index=False)
# df["epoch_batch"] = df["epoch"].astype(str) + "-" + df["batch"].astype(str)
# sns.lineplot(data=df, x="epoch_batch", y="Acc", hue="Set")
# plt.title("Train/ Test Set Accuracy")
# plt.xlabel("Epoch-Batch")
# plt.ylabel("Accuracy")
# plt.show()
