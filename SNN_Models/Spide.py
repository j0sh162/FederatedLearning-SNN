import os
import sys
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import copy
from modules.snn_spide_module import SNNSPIDEModule
from modules.snn_modules import SNNIFFuncMultiLayer, SNNLIFFuncMultiLayer, SNNConv
from utils import AverageMeter,accuracy
import time




class SNNSPIDEConvNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(SNNSPIDEConvNet, self).__init__()
        self.parse_cfg(cfg)
    
        self.network_x = SNNConv(self.c_in, self.c_hidden, self.kernel_size_x, bias=True, stride=self.stride_x, padding=self.padding_x, dropout=self.dropout, init='kaiming')
        network_s = SNNConv(self.c_hidden, self.c_hidden, self.kernel_size_s, bias=False, stride=1, dropout=self.dropout, init='kaiming')
        # network_s._wnorm(norm_range=10.)
        
        self.network_s = network_s
        if self.leaky == None:
            self.snn_func = SNNIFFuncMultiLayer(nn.ModuleList([self.network_s]), self.network_x, vth=self.vth, vth_back=self.vth_back, u_rest=self.u_rest, u_rest_back=self.u_rest_back)
        else:
            self.snn_func = SNNLIFFuncMultiLayer(nn.ModuleList([self.network_s]), self.network_x, vth=self.vth, vth_back=self.vth_back, u_rest=self.u_rest, u_rest_back=self.u_rest_back, leaky=self.leaky)

        self.snn_spide_conv = SNNSPIDEModule(self.snn_func)

        self.classifier = nn.Linear(self.h_hidden * self.w_hidden * self.c_hidden, self.num_classes, bias=True)

    def parse_cfg(self, cfg):
        self.c_in = cfg['MODEL']['c_in']
        self.c_hidden = cfg['MODEL']['c_hidden']
        self.h_hidden = cfg['MODEL']['h_hidden']
        self.w_hidden = cfg['MODEL']['w_hidden']
        self.num_classes = cfg['MODEL']['num_classes']
        self.kernel_size_x = cfg['MODEL']['kernel_size_x']
        self.stride_x = cfg['MODEL']['stride_x']
        self.padding_x = cfg['MODEL']['padding_x']
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

        x1 = torch.zeros([B, self.c_hidden, self.h_hidden, self.w_hidden]).to(dev)
        self.snn_func.network_x._reset(x1)
        self.snn_func.network_s_list[0]._reset(x1)

        z = torch.zeros([B, self.c_hidden, H, W]).to(dev)
        z = self.snn_spide_conv(z, x, time_step=time_step, time_step_back=time_step_back, input_type=input_type, leaky=leaky)

        return z

    def forward(self, x, **kwargs):
        B = x.size(0)
        z = self._forward(x, **kwargs)
        z = z.reshape(B, -1)
        y = self.classifier(z)

        return y

    def get_forward_firing_rate(self):
        firing_rate = self.snn_spide_conv.snn_func.last_forward_firing_rate
        firing_rate = firing_rate[0]
        B = firing_rate.shape[0]
        firing_rate = firing_rate.reshape(B, -1)
        return torch.sum(torch.mean(firing_rate, dim=1))

    def get_backward_firing_rate(self):
        firing_rate = self.snn_spide_conv.snn_func.last_backward_firing_rate
        firing_rate = firing_rate[0]
        B = firing_rate.shape[0]
        firing_rate = firing_rate.reshape(B, -1)
        return torch.sum(torch.mean(firing_rate, dim=1))

def train(model,trainloader,device, epochs, optimizer,warmup=0):
    # switch to train mode
    model.train()
    criterion = nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    epoch_losses = []
    epoch_accs = []


    average_forward_firing_rate = 0.
    average_backward_firing_rate = 0.
    batch_num = 0

    for current_iter in range(epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if warmup != 0 and current_iter < warmup:
                adjust_warmup_lr(optimizer, current_iter, warmup)
                current_iter += 1
            # measure data loading time
            data_time.update(time.time() - end)

            # change the dimension of inputs from B*T*C*H*W to B*C*H*W*T
            inputs = inputs.permute(1, 2, 3, 4, 0)
            # take the first T time steps
            inputs = inputs[:, :, :, :, :model.time_step]

            if device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs, input_type='spikes')
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            scaled_loss = loss * 400
            optimizer.zero_grad()
            #loss.backward()
            scaled_loss.backward()
            optimizer.step()

            # get firing rate statistics
            forward_firing_rate = model.get_forward_firing_rate()
            backward_firing_rate = model.get_backward_firing_rate()
            average_forward_firing_rate += forward_firing_rate
            average_backward_firing_rate += backward_firing_rate
            batch_num += inputs.shape[0]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print(f'loss: {loss}, Accuarcy:{top1.avg}')

       
    return (losses.avg, top1.avg, average_forward_firing_rate/batch_num, average_backward_firing_rate/batch_num)


def test(testloader, model, device):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()

    end = time.time()


    average_forward_firing_rate = 0.
    batch_num = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # change the dimension of inputs from B*T*C*H*W to B*C*H*W*T
        inputs = inputs.permute(1, 2, 3, 4, 0)
        # take the first T time steps
        inputs = inputs[:, :, :, :, :model.time_step]

        if device == 'cuda':
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs, input_type='spikes')
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # get firing rate statistics
        forward_firing_rate = model.get_forward_firing_rate()
        average_forward_firing_rate += forward_firing_rate
        batch_num += inputs.shape[0]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress

    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [50]:
        state['lr'] *= 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr'] / state['scale_factor']

def adjust_warmup_lr(optimizer, citer, warmup):
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr'] / state['scale_factor'] * (citer + 1.) / warmup
