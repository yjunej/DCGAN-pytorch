import torchvision
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import os

def visualization_grid(image, size=(1,64,64), num=36, step=0):
    '''
    image: torch.tensor shape like (num_of_image, channels, width, height)
    size: channels, width, height
    num: number of display images
    '''
    img = image[:num]
    img = img.cpu().detach()
    grid = torchvision.utils.make_grid(img,nrow=6)
    plt.imshow(grid.permute(1,2,0).squeeze())
    plt.title('Epoch {}'.format(step//485+1),fontdict = {'fontsize' : 80})
    plt.savefig(os.path.join('./result',str(step).zfill(5)+'.png'))
    plt.show()

def noise_maker(batch_size=128,z_dim=100,device='cpu'):
    '''
    batch_size: batch size
    z_dim: noise vector dimension
    device: device
    '''
    return torch.randn(batch_size,z_dim,device=device).reshape(batch_size,z_dim,1,1)

def weight_initialize(m):
    '''
    m: pytorch Module

    All weights are initialized from a zero-centered Normal distribution with standard deviation 0.02
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)

def make_result_dir(path='.'):
    os.mkdir(path+'result')