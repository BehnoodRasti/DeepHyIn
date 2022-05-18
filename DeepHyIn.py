#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:38:32 2022

@author: behnood
"""

from __future__ import print_function
import matplotlib.pyplot as plt
#matplotlib inline

import os
from common import Concat
from common import bn
from common import act
#from skimage.metrics import peak_signal_noise_ratio
#from skimage.measure import compare_psnr
#from skimage.measure import compare_mse
import numpy as np
import scipy.io
import UnmixArch
import torch
import torch.optim
import torch.nn as nn
from UtilityMine import *
#from utils.inpainting_utils import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
#%%
#file_name  = 'data/inpainting/inpainting1925.mat'
# file_name  = '/home/nvidia/OneDrive/DeepLearning/HSI Denoising/deep-hs-prior-master/data/inpainting/inpainting_dc.mat'
file_name  = "Samson/Y_clean.mat"
mat = scipy.io.loadmat(file_name)
img_np = mat["Y_clean"]
img_np = img_np.transpose(2,0,1)
[p1, nr1, nc1] = img_np.shape
img_var = torch.from_numpy(img_np).type(dtype)
file_name2  = "Samson/mask.mat"
mat2 = scipy.io.loadmat(file_name2)
mask_np = mat2["mask"]
mask_np = np.expand_dims(mask_np,2)
mask_np = mask_np.transpose(2,0,1)
mask_var = torch.from_numpy(mask_np).type(dtype)
# ensure dimensions [0][1] are divisible by 32 (or 2^depth)!
#%%
fname3  = "Samson/A_true.mat"
# ensure dimensions [0][1] are divisible by 32 (or 2^depth)!.mat"
# ensure dimensions [0][1] are divisible by 32 (or 2^depth)!
mat3 = scipy.io.loadmat(fname3)
A_true_np = mat3["A_true"]
A_true_np = A_true_np.transpose(2,0,1)
A_true=A_true_np*mask_np
#%%
fname4  = "Samson/E.mat"
#fname4 = "/home/nvidia/OneDrive/Unmixing_Paper/Result/Sim/Blind/TruEnd/TV/New2/Eest"
#fname4  = "/home/nvidia/OneDrive/Unmixing_Paper/Result/Sam/Blind/TruEnd/TV/New2/E_est.mat" 
mat4 = scipy.io.loadmat(fname4)
E_np = mat4["E"]
n_lin=50
E_Lin=np.zeros((1,n_lin))
E_Lin[0,:]=np.linspace(0.5,1.5,n_lin)
rmax=E_np.shape[1]
EE=np.zeros((p1,rmax*n_lin))
#%%

# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
# band = 110
# ax1.imshow(img_var.detach().cpu().numpy()[band,:,:], cmap='gray')
# ax2.imshow(mask_np.squeeze(), cmap='gray') 
# ax3.imshow((img_var*mask_var).detach().cpu().numpy()[band,:,:], cmap='gray')
# plt.show()
#%%


# class Lin(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(Lin, self).__init__(
#             in_channels, out_channels)
#     def Lin_forward(self, input, in_channels, out_channels):
#                 x3 = torch.transpose(input.view((rmax,nr1*nc1)),0,1) 
#                 # x4 = torch.transpose(x4,0,1)
#                 # x4=x4.view((1,p1,nr1,nc1))
#                 return nn.Linear(x3,self.in_features,self.out_features,bias)
#         # if self.padding_mode == 'circular':
#         #     expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#         #                         (self.padding[0] + 1) // 2, self.padding[0] // 2)
#         #     return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#         #                     weight, self.bias, self.stride,
#         #                     _pair(0), self.dilation, self.groups)
#         # return F.conv2d(input, weight, self.bias, self.stride,
#         #                 self.padding, self.dilation, self.groups)

#     # def forward(self, input):
#     #     return self.Lin_forward(input, self.weight)
def UnmixArch(
        num_input_channels=2, num_output_channels=3, output=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        UnmixArch = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, UnmixArch, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            UnmixArch.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            UnmixArch.add(bn(num_channels_skip[i]))
            UnmixArch.add(act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, align_corners=True,mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
#      model.add(nn.ReLU())
       # model.add(nn.Sigmoid())
    #    model.add(nn.Linear(num_output_channels,num_output_channels))
    #    model.add(nn.Hardtanh(min_val=0,max_val=1))
    #    model.add(act('ASC'))
        model.add(nn.Softmax(dim=1))
    #    model.add(conv(num_output_channels, output, 1, bias=need_bias, pad=pad))
    return model
#%%
pad = 'reflection' #'zero'
OPT_OVER = 'net'
OPTIMIZER = 'adam'

# method = '2D'
INPUT = 'meshgrid' #'noise'# 
# input_depth = img_np.shape[0] 
LR = 0.01  
num_iter = 20001

show_every = 500
input_depth = 2#img_np.shape[0]

class CAE_AbEst(nn.Module):
    def __init__(self):
        super(CAE_AbEst, self).__init__()
        # encoding layers
        self.conv1 = nn.Sequential(
            UnmixArch(
                    input_depth, rmax,
                    # num_channels_down = [8, 16, 32, 64, 128], 
                    # num_channels_up   = [8, 16, 32, 64, 128],
                    # num_channels_skip = [4, 4, 4, 4, 4], 
                    num_channels_down = [128] * 5,
                    num_channels_up =   [128] * 5,
                    num_channels_skip =    [ 4] * 5,  
                    filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
                    upsample_mode='bilinear', # downsample_mode='avg',
                    need1x1_up=True,
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

net = CAE_AbEst()
net.cuda()
print(net)

net = net.type(dtype)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
#%%
# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

# Loss
# mse = torch.nn.MSELoss().type(dtype)
def my_loss(target, outLR, mask_var, End):
    HR=torch.mm(End.view(p1,rmax),outLR.view(rmax,nr1*nc1))
    loss1 = 0.5*torch.norm((HR.view(1,p1,nr1,nc1)*mask_var- target), 'fro')**2
    return loss1
E_torch = torch.from_numpy(E_np).type(dtype)
img_var = img_var[None, :].cuda()
mask_var = mask_var[None, :].cuda()
# #%%
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
# ax1.imshow(img_var.detach().cpu().numpy()[0,10,:,:], cmap='gray')
# ax2.imshow(img_var.detach().cpu().numpy()[0,60,:,:], cmap='gray')
# ax3.imshow(img_var.detach().cpu().numpy()[0,90,:,:], cmap='gray')
# plt.show()
#%%        
i = 0
def closure():
    
    global i,out_np, out_HR_np
         
    out = net(net_input)
    out_HR=torch.mm(E_torch,out.view(rmax,nr1*nc1))
    total_loss = my_loss( img_var * mask_var, out, mask_var,E_torch)
    total_loss.backward()
    out_np = out.detach().cpu().numpy()[0]
    out_HR_np_sub = out_HR.detach().cpu().squeeze().numpy()
    out_HR_np=out_HR_np_sub.reshape((p1,nr1,nc1))                
    if  i % show_every == 0:
        out_np = out.detach().cpu().numpy()[0]
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
        ax1.imshow(np.stack((out_np[2,:,:],out_np[1,:,:],out_np[0,:,:]),2))
        ax2.imshow(np.stack((A_true[2,:,:],A_true[1,:,:],A_true[0,:,:]),2))
        ax3.imshow(np.stack((A_true_np[2,:,:],A_true_np[1,:,:],A_true_np[0,:,:]),2))
        plt.show()             
    i += 1
    return total_loss

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)