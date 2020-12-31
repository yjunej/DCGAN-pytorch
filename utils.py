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
plt.rcParams["figure.figsize"] = (24,24)

def visualization_grid(image, size=(1,64,64), num=36,step=0, save=True,nrow=None, epochs=0, denorm=False):
    '''
    Show grid image

    Inputs: 
        image: torch.tensor shape like (num_of_image, channels, width, height)
        size: channels, width, height
        num: number of display images
        step: batch step for file name
        save: True if save results
        nrow: number of grid row
        epochs: current epochs for result image title
    
    '''
    img = image[:num]
    if denorm:
        img = img*0.5 + 0.5
    img = img.cpu().detach()
    if nrow==None:
        nrow = int(math.sqrt(num))
    grid = torchvision.utils.make_grid(img,nrow=nrow)
    plt.imshow(grid.permute(1,2,0).squeeze())
    if save:
        plt.title('Epoch {}'.format(epochs+1),fontdict = {'fontsize' : 80})
        plt.savefig(os.path.join('./result',str(step).zfill(5)+'.png'))
    # plt.show()


def interpolate_noise(starts,end,num,device):
    '''
    interpolate noise vectors from starts to end
    
    inputs: 
        starts: noise vectors list ([noise1, noise2], noise.shape=(100,1,1))
        end: A noise vecotr (target_noise.shape=(100,1,1))
        num: num of interpolation output
        device: device for processing
    
    outputs:
        interpolation torch tensor from each noise in starts to end noise
    '''
    s = []
    
    for sn in starts:
        s.append(sn.cpu().detach())
    e = end.cpu().detach()
    interpolate_matrix = np.linspace(s[0],e,num)
    for x in s[1:]:
        interpolate_matrix=np.concatenate([interpolate_matrix,np.linspace(x,e,num)],axis=0)
   
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
    All weights are initialized from a zero-centered Normal distribution with standard deviation 0.02

    Inputs:
        m: pytorch Module

    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)

def make_result_dir(path='.'):
    '''
    Create directory for saving result images
    '''
    try:
        os.mkdir(path+'/result')
        print('Result folder has created!'.center(60,'='))
    except:
        print('Result folder already exists, skip creating directory'.center(60,'='))