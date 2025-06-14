# Based on: https://ieeexplore.ieee.org/abstract/document/10242251
# Adapted from: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html
import snntorch as snn
import torch
import torch.nn as nn
from rich import print
from snntorch import surrogate, utils

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


class Net(nn.Module):
    def __init__(
        self,
        input_shape: tuple = (2, 34, 34),
        num_hidden: int = 1,
        num_output: int = 10,
        spike_grad=surrogate.atan(),
        beta: float = 0.9,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.spike_grad = spike_grad
        self.beta = beta

        # self.net = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=12, kernel_size=5),  # 12x30x30
        #     nn.MaxPool2d(kernel_size=2),  # 12x15x15
        #     snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
        #     nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5),  # 32x11x11
        #     nn.MaxPool2d(kernel_size=2),  # 32x5x5
        #     snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
        #     nn.Flatten(),
        #     nn.Linear(in_features=32 * 5 * 5, out_features=10),
        #     snn.Leaky(
        #         beta=self.beta, spike_grad=spike_grad, init_hidden=True, output=True
        #     ),
        # ).to(device)

        # Network based on: https://doi.org/10.48550/arXiv.2105.08810
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(in_features=32 * 15 * 15, out_features=10),
            snn.Leaky(
                beta=self.beta, spike_grad=spike_grad, init_hidden=True, output=True
            ),
        ).to(device)

    def forward(self, x):
        spk_rec = []
        x.to(device)

        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        for step in range(x.size(0)):  # data.size(0) = number of time steps
            # print("x shape", x.shape)
            # print("x step", x[step].shape)
            spk_out, mem_out = self.net(x[step])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)
