import torchvision
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import math 

def visualization_grid(image, size=(1,64,64), num=36,step=0, save=True,nrow=None):
    '''
    image: torch.tensor shape like (num_of_image, channels, width, height)
    size: channels, width, height
    num: number of display images
    '''
    img = image[:num]
    img = img.cpu().detach()
    if nrow==None:
        nrow = int(math.sqrt(num))
    grid = torchvision.utils.make_grid(img,nrow=nrow)
    plt.imshow(grid.permute(1,2,0).squeeze())
    if save:
        plt.title('Epoch {}'.format(step//485+1),fontdict = {'fontsize' : 80})
        plt.savefig(os.path.join('./img',str(step).zfill(5)+'.png'))
    plt.show()


def interpolate_noise(starts,end,num,device):
    interpolate_matrix = np.linspace(starts[0],end,num)
    for x in starts[1:]:
        interpolate_matrix=np.concatenate([interpolate_matrix,np.linspace(x,end,num)],axis=0)
   
    return torch.tensor(interpolate_matrix,device=device)

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