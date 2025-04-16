# Based on: https://ieeexplore.ieee.org/abstract/document/10242251
# Adapted from: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html
import matplotlib.pyplot as plt
import nir
import snntorch as snn
import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
import torchvision
from rich import print
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import surrogate, utils
from snntorch.export_nir import export_to_nir
from tonic import DiskCachedDataset, MemoryCachedDataset
from torch.utils.data import DataLoader

from FederatedSNN.SNN import SNN

sensor_size = tonic.datasets.NMNIST.sensor_size
sensor_size = tonic.datasets.CIFAR10DVS.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose(
    [
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ]
)

trainset = tonic.datasets.NMNIST(
    save_to="./data", transform=frame_transform, train=True
)
testset = tonic.datasets.NMNIST(
    save_to="./data", transform=frame_transform, train=False
)

transform = tonic.transforms.Compose(
    [torch.from_numpy, torchvision.transforms.RandomRotation([-10, 10])]
)

cached_trainset = MemoryCachedDataset(trainset, transform=transform)

# no augmentations for the testset
cached_testset = MemoryCachedDataset(testset)

batch_size = 128
train_loader = DataLoader(
    cached_trainset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=True,
)
test_loader = DataLoader(
    cached_testset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

snn_net = SNN.Net(
    input_shape=None,
    num_hidden=None,
    num_output=None,
    spike_grad=surrogate.atan(),
    beta=0.5,
)

print("Sensor size:", sensor_size)
print(snn_net.net)
# Print the shape of tensors at each step in the nn.Sequential
x_dummy = torch.randn(10, 2, 34, 34).to(device)  # Example input tensor
print(f"Input shape: {x_dummy.shape}")

for layer in snn_net.net:
    x_dummy = layer(x_dummy)
    if isinstance(x_dummy, tuple):  # Handle layers that return (spike, membrane)
        x_dummy = x_dummy[0]
    print(f"After {layer.__class__.__name__}: {x_dummy.shape}")
quit()

optimizer = torch.optim.Adam(snn_net.net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 1
num_iters = 20
num_batches = len(train_loader)

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    for batch_number, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        snn_net.net.train()
        spk_rec = snn_net.forward(data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)

        print(
            "Epoch {:02d} | Batch {:03d}/{:03d} | Loss {:05.2f} | Accuracy {:05.2f}%".format(
                epoch, batch_number, num_batches, loss_val.item(), acc * 100
            )
        )

        # training loop breaks after num_iters iterations/ batches
        if batch_number == num_iters:
            break

quit()

# Export to NIR and add example data and output
# TODO NIR Export not working at the moment
print(data[0].shape)
print(targets[0].shape)
print(spk_rec[0].shape)
nir_graph = export_to_nir(snn_net.cpu(), sample_data=torch.randn(1, 2, 34, 34).cpu())
nir.write(filename="example.nir", graph=nir_graph)

# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
