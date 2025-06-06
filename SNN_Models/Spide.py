import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch._utils
import torch.nn.functional as F
import copy
from modules.snn_spide_module import SNNSPIDEModule
from modules.snn_modules import SNNIFFuncMultiLayer, SNNLIFFuncMultiLayer, SNNConv, SNNFC, SNNZero, SNNPooling
from utils import AverageMeter,accuracy
logger = logging.getLogger(__name__)


class SNNSPIDEConvMultiLayerNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(SNNSPIDEConvMultiLayerNet, self).__init__()
        self.parse_cfg(cfg)

        self.network_x = SNNConv(self.c_in, self.c_hidden, self.kernel_size_x, bias=True, stride=1, padding=self.padding_x, dropout=0.0, init='kaiming')
        self.network_p1 = SNNPooling()

        self.network_s1 = SNNConv(self.c_hidden, self.c_s1, self.kernel_size_s, bias=True, stride=1, dropout=0.0, init='kaiming')
        self.network_p2 = SNNPooling()
        
        h_after_conv_pool = (self.h_hidden // self.stride_x // 2 // 2)
        w_after_conv_pool = (self.w_hidden // self.stride_x // 2 // 2)
        fc_input_features = h_after_conv_pool * w_after_conv_pool * self.c_s1
        
        self.network_s2 = SNNFC(fc_input_features, self.fc_num, bias=True, need_resize=True, sizes=[1, self.fc_num], dropout=0.5, init='kaiming')

        snn_zero_h = self.h_hidden // self.stride_x
        snn_zero_w = self.w_hidden // self.stride_x

        self.network_s3 = SNNZero(input_size=[1, self.fc_num], output_size=[1, self.c_hidden, snn_zero_h, snn_zero_w])

        if self.leaky == None:
            self.snn_func = SNNIFFuncMultiLayer(nn.ModuleList([self.network_p1, self.network_s1, self.network_p2, self.network_s2, self.network_s3]), self.network_x, vth=self.vth, vth_back=self.vth_back, u_rest=self.u_rest, u_rest_back=self.u_rest_back)
        else:
            self.snn_func = SNNLIFFuncMultiLayer(nn.ModuleList([self.network_p1, self.network_s1, self.network_p2, self.network_s2, self.network_s3]), self.network_x, vth=self.vth, leaky=self.leaky, vth_back=self.vth_back, u_rest=self.u_rest, u_rest_back=self.u_rest_back)

        self.snn_spide_conv = SNNSPIDEModule(self.snn_func)
        self.classifier = nn.Linear(self.fc_num, self.num_classes, bias=True)

    def parse_cfg(self, cfg):
        self.c_in = cfg['MODEL']['c_in']
        self.c_hidden = cfg['MODEL']['c_hidden']
        self.c_s1 = cfg['MODEL']['c_s1']
        self.h_hidden = cfg['MODEL']['h_hidden']
        self.w_hidden = cfg['MODEL']['w_hidden']
        self.fc_num = cfg['MODEL']['fc_num']
        self.num_classes = cfg['MODEL']['num_classes']
        self.kernel_size_x = cfg['MODEL']['kernel_size_x']
        self.stride_x = cfg['MODEL']['stride_x']
        self.padding_x = cfg['MODEL']['padding_x']
        self.pooling_x = cfg['MODEL']['pooling_x'] if 'pooling_x' in cfg['MODEL'].keys() else False
        self.kernel_size_s = cfg['MODEL']['kernel_size_s']
        self.time_step = cfg['MODEL']['time_step']
        self.time_step_back = cfg['MODEL']['time_step_back']
        self.vth = cfg['MODEL']['vth']
        self.vth_back = cfg['MODEL']['vth_back'] if 'vth_back' in cfg['MODEL'].keys() else self.vth
        self.u_rest = cfg['MODEL']['u_rest'] if 'u_rest' in cfg['MODEL'].keys() else None
        self.u_rest_back = cfg['MODEL']['u_rest_back'] if 'u_rest_back' in cfg['MODEL'].keys() else None
        self.dropout = cfg['MODEL']['dropout'] if 'dropout' in cfg['MODEL'].keys() else 0.0
        self.leaky = cfg['MODEL']['leaky'] if 'leaky' in cfg['MODEL'].keys() else None

    def _forward(self, x, **kwargs):
        time_step = kwargs.get('time_step', self.time_step)
        time_step_back = kwargs.get('time_step_back', self.time_step_back)
        input_type = kwargs.get('input_type', 'constant')
        leaky = kwargs.get('leaky', self.leaky)
        dev = x.device

        if input_type == 'constant':
            B, C, H, W = x.size()
        else:
            B, C, H, W, _ = x.size()

        x1 = torch.zeros([B, self.c_hidden, H//self.stride_x, W//self.stride_x]).to(x.device)
        self.snn_func.network_x._reset(x1)

        x1 = torch.zeros([B, self.c_s1, H//(self.stride_x*2), W//(self.stride_x*2)]).to(x.device)
        self.network_s1._reset(x1)

        x1 = torch.zeros([B, self.fc_num]).to(x.device)
        self.network_s2._reset(x1)

        z = torch.zeros([B, self.c_hidden, H//2, W//2]).to(x.device)
        z = self.snn_spide_conv(z, x, time_step=time_step, time_step_back=time_step_back, input_type=input_type, leaky=leaky)

        return z

    def forward(self, x, **kwargs):
        B = x.size(0)
        z = self._forward(x, **kwargs)
        z = z.reshape(B, -1)
        y = self.classifier(z)

        return y

    def get_forward_firing_rate(self):
        firing_rates = self.snn_spide_conv.snn_func.last_forward_firing_rate
        num = 0.
        rate = 0.
        B = firing_rates[0].shape[0]
        for firing_rate in firing_rates:
            firing_rate = firing_rate.reshape(B, -1)
            num += firing_rate.shape[1]
            rate += torch.sum(firing_rate)

        return rate / num

    def get_backward_firing_rate(self):
        firing_rates = self.snn_spide_conv.snn_func.last_backward_firing_rate
        num = 0.
        rate = 0.
        B = firing_rates[0].shape[0]
        for firing_rate in firing_rates:
            firing_rate = firing_rate.reshape(B, -1)
            num += firing_rate.shape[1]
            rate += torch.sum(firing_rate)

        return rate / num

def train_fl(
    model,
    trainloader,
    device,
    epochs,
    optimizer,
):
    model.train()
    criterion = nn.CrossEntropyLoss()


    epoch_losses = []
    epoch_accs = []

    for epoch in range(epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(iter(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            # NMNIST with ToFrame and PadTensors(batch_first=False) gives (T, B, C, H, W)
            # If your SNN expects (B, C, H, W) and processes time internally,
            # you might need to reshape or select frames.
            # Example: if model processes one frame at a time and averages outputs:
            # B, T, C, H, W = inputs.shape # if batch_first=True
            # T, B, C, H, W = inputs.shape # if batch_first=False (current case)
            # For simplicity, let's assume the model's forward can handle (T,B,C,H,W)
            # or you adapt it. If it expects (B,C,H,W), you might do:
            # inputs = inputs.mean(dim=0) # Average over time, or select first frame inputs[0]
            # This part is CRITICAL and depends on your SNNSPIDEConvMultiLayerNet's forward pass.

            outputs = model(inputs) # Model handles its internal time_steps
            loss = criterion(outputs, targets)

            prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5)) # Ensure accuracy fn is compatible
            losses.update(loss.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0)) # inputs.size(1) is batch size if T,B,C,H,W
            top1.update(prec1.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0))

            scaled_loss = loss * 100.0
            optimizer.zero_grad()
            scaled_loss.backward()
            optimizer.step()
        print(f"  Client Epoch {epoch+1}/{epochs} -> Loss: {losses.avg:.4f}, Acc: {top1.avg:.2f}%")
        epoch_losses.append(losses.avg)
        epoch_accs.append(top1.avg)
    return np.mean(epoch_losses), np.mean(epoch_accs)

# TODO Update Spides forward for NMNIST

# Modified test function for Flower client and server-side evaluation
# def test_fl(model, testloader, device, args_dict):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     with torch.no_grad():
#         for inputs, targets in testloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs) # Assuming model handles time_steps internally
#             loss = criterion(outputs, targets)

#             prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
#             losses.update(loss.item(), inputs.size(0))
#             top1.update(prec1.item(), inputs.size(0))
#     return losses.avg, top1.avg

def test_fl(model, testloader, device): # args_dict might not be needed if model config is fixed
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Similar consideration for input shape as in train_fl
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0))
            top1.update(prec1.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0))
    return losses.avg, top1.avg