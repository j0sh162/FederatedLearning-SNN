import tonic
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
device = "cuda" if torch.cuda.is_available() else "cpu"

class WrapperFunction(Function):
    @staticmethod
    def forward(ctx, input, params, forward, backward):
        ctx.backward = backward
        pack, output = forward(input)
        ctx.save_for_backward(*pack)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        backward = ctx.backward
        pack = ctx.saved_tensors
        grad_input, grad_weight = backward(grad_output, *pack)
        return grad_input, grad_weight, None, None

class FirstSpikeTime(Function):
    @staticmethod
    def forward(ctx, input):   
        idx = torch.arange(input.shape[2], 0, -1).unsqueeze(0).unsqueeze(0).float()
        first_spike_times = torch.argmax(idx*input, dim=2).float()
        ctx.save_for_backward(input, first_spike_times.clone())
        first_spike_times[first_spike_times==0] = input.shape[2]-1
        return first_spike_times

    @staticmethod
    def backward(ctx, grad_output):
        input, first_spike_times = ctx.saved_tensors
        k = F.one_hot(first_spike_times.long(), input.shape[2]).float()
        grad_input = k * grad_output.unsqueeze(-1)
        return grad_input

class SpikingLinear(nn.Module):
    def __init__(self, input_dim, output_dim, T, dt, tau_m, tau_s, mu):
        super(SpikingLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = T
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.normal_(self.weight, mu, mu)

        self.forward = lambda input : WrapperFunction.apply(input, self.weight, self.manual_forward, self.manual_backward)

    def manual_forward(self, input):
        steps = int(self.T / self.dt)

        V = torch.zeros(input.shape[0], self.output_dim, steps)
        I = torch.zeros(input.shape[0], self.output_dim, steps)
        output = torch.zeros(input.shape[0], self.output_dim, steps)

        while True:
            for i in range(1, steps):
                t = i * self.dt
                V[:,:,i] = (1 - self.dt / self.tau_m) * V[:,:,i-1] + (self.dt / self.tau_m) * I[:,:,i-1]
                I[:,:,i] = (1 - self.dt / self.tau_s) * I[:,:,i-1] + F.linear(input[:,:,i-1].float(), self.weight)
                spikes = (V[:,:,i] > 1.0).float()
                output[:,:,i] = spikes
                V[:,:,i] = (1-spikes) * V[:,:,i]

            if self.training:
                is_silent = output.sum(2).min(0)[0] == 0
                self.weight.data[is_silent] = self.weight.data[is_silent] + 1e-1
                if is_silent.sum() == 0:
                    break
            else:
                break

        return (input, I, output), output

    def manual_backward(self, grad_output, input, I, post_spikes):
        steps = int(self.T / self.dt)

        lV = torch.zeros(input.shape[0], self.output_dim, steps)
        lI = torch.zeros(input.shape[0], self.output_dim, steps)

        grad_input = torch.zeros(input.shape[0], input.shape[1], steps)
        grad_weight = torch.zeros(input.shape[0], *self.weight.shape)

        for i in range(steps-2, -1, -1):
            t = i * self.dt
            delta = lV[:,:,i+1] - lI[:,:,i+1]
            grad_input[:,:,i] = F.linear(delta, self.weight.t())
            lV[:,:,i] = (1 - self.dt / self.tau_m) * lV[:,:,i+1] + \
                post_spikes[:,:,i+1] * (lV[:,:,i+1] + grad_output[:,:,i+1]) / (I[:,:,i] - 1 + 1e-10)
            lI[:,:,i] = lI[:,:,i+1] + (self.dt / self.tau_s) * (lV[:,:,i+1] - lI[:,:,i+1])
            spike_bool = input[:,:,i].float()
            grad_weight -= (spike_bool.unsqueeze(1) * lI[:,:,i].unsqueeze(2))

        return grad_input, grad_weight

class SNN(nn.Module):
    def __init__(self, input_dim, output_dim, T, dt, tau_m, tau_s,xi,alpha,beta):
        super(SNN, self).__init__()
        self.slinear1 = SpikingLinear(input_dim, output_dim, T, dt, tau_m, tau_s, 0.1)
        # self.slinear2 = SpikingLinear(100, output_dim, T, dt, tau_m, tau_s, 0.1)
        self.outact = FirstSpikeTime.apply
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.tau_s = tau_s
        self.criterion = SpikeCELoss(T,xi,tau_s)
        

    def forward(self, input):
        u = self.slinear1(input)
        # u = self.slinear2(u)
        u = self.outact(u)
        return u

class SpikeCELoss(nn.Module):
    # TODO change to also return accuarcy 
    def __init__(self, T, xi, tau_s):
        super(SpikeCELoss, self).__init__()
        self.xi = xi
        self.tau_s = tau_s
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.celoss(-input / (self.xi * self.tau_s), target)
        return loss

def train(model, optimizer, loader,epochs,device):
    total_correct = 0.
    total_loss = 0.
    total_samples = 0.
    model.train()
    alpha = model.alpha
    beta = model.beta
    tau_s = model.tau_s
    T = model.T
    criterion = model.criterion
    # TODO add epochs 


    total_correct = 0.
    total_loss = 0.
    total_samples = 0.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epochs):
        for batch_idx, (input, target) in enumerate(iter(loader)):
            input, target = input.to(device), target.to(device)
            index = 0
            if input.shape[0] == T:
                index = 1
                
            input = input.view(input.shape[index], -1, T)

            output = model(input)

            loss = criterion(output, target)

            if alpha != 0:
                target_first_spike_times = output.gather(1, target.view(-1, 1))
                loss += alpha * (torch.exp(target_first_spike_times / (beta * tau_s)) - 1).mean()

            predictions = output.data.min(1, keepdim=True)[1]
            total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
            total_loss += loss.item() * len(target)
            total_samples += len(target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # print updates every 10 batches
            if batch_idx % 10 == 0:
                print('\tBatch {:03d}/{:03d}: \tAcc {:.2f}  Loss {:.3f}'.format(
                    batch_idx, len(loader), 100*total_correct/total_samples, total_loss/total_samples
                ))

        print('\t\tTrain: \tAcc {:.2f}  Loss {:.3f}'.format(100*total_correct/total_samples, total_loss/total_samples))
        scheduler.step()
    return total_loss/total_samples,100*total_correct/total_samples

def test(model, loader, device):
    total_correct = 0.
    total_samples = 0.
    total_loss = 0.
    model.eval()

    alpha = model.alpha
    beta = model.beta
    tau_s = model.tau_s
    T = model.T
    criterion = model.criterion
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            index = 0
            if data.shape[0] == T:
                index = 1
                
            spike_data = data.view(data.shape[index], -1, T)

            first_post_spikes = model(spike_data)
            loss = criterion(first_post_spikes, target)
            
            # if alpha != 0:
            #     target_first_spike_times = spike_data.gather(1, target.view(-1, 1))
            #     loss += alpha * (torch.exp(target_first_spike_times / (beta * tau_s)) - 1).mean()
                
            predictions = first_post_spikes.data.min(1, keepdim=True)[1]
            total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
            total_samples += len(target)
            total_loss += loss.item() * len(target)

            # print updates every 10 batches
            if batch_idx % 10 == 0:
                print('\tBatch {:03d}/{:03d}: \tAcc {:.2f}'.format(batch_idx, len(loader), 100*total_correct/total_samples))

        print('\t\tTest: \tAcc {:.2f}'.format(100*total_correct/total_samples))
    return total_loss/total_samples,100*total_correct/total_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument('--data-folder', type=str, default='data', help='name of folder to place dataset (default: data)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--deterministic', action='store_true', help='run in deterministic mode for reproducibility')

    # training settings
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate (default: 1.0)')
    parser.add_argument('--batch-size', type=int, default=128, help='size of batch used for each update step (default: 128)')

    # loss settings (specific for SNNs)
    parser.add_argument('--xi', type=float, default=0.4, help='constant factor for cross-entropy loss (default: 0.4)')
    parser.add_argument('--alpha', type=float, default=0.01, help='regularization factor for early-spiking (default: 0.01)')
    parser.add_argument('--beta', type=float, default=2, help='constant factor for regularization term (default: 2.0)')

    # SNN settings
    parser.add_argument('--T', type=float, default=20, help='duration for each simulation, in ms (default: 20)')
    parser.add_argument('--dt', type=float, default=1, help='time step to discretize the simulation, in ms (default: 1)')
    parser.add_argument('--tau_m', type=float, default=20.0, help='membrane time constant, in ms (default: 20)')
    parser.add_argument('--tau_s', type=float, default=5.0, help='synaptic time constant, in ms (default: 5)')

    args = parser.parse_args() 

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    transform = tonic.transforms.Compose(
        [
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=args.T),
        ]
    )

    train_dataset = tonic.datasets.NMNIST(args.data_folder, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = tonic.datasets.NMNIST(args.data_folder, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2 (C) * 34 (W) * 34 (H) = 2312 input neurons
    model = SNN(2312, 10, args.T, args.dt, args.tau_m, args.tau_s,args.xi,0.01,2).to(device)
    # criterion = SpikeCELoss(args.T, args.xi, args.tau_s)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(args.epochs):
        print('Epoch {:03d}/{:03d}'.format(epoch, args.epochs))
        train(model, optimizer, train_loader, 1,device)
        test(model, test_loader, device)
        # scheduler.step()
