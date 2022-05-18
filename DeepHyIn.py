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
from common import *

#from skimage.metrics import peak_signal_noise_ratio
#from skimage.measure import compare_psnr
#from skimage.measure import compare_mse
import numpy as np
import scipy.io
from UnmixArch import UnmixArch
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
