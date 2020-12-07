from utils import *

class DCGAN_Generator(nn.Module):
    '''
    z_dim: noise space dimension
    h_dim: last hidden layer dimension
    channels: data channels (RGB:3, GRAYSCALE: 1)

    '''
    def __init__(self, z_dim=100, h_dim=128, channels=1):
        super(DCGAN_Generator, self).__init__()
        self.convt0 = nn.ConvTranspose2d(in_channels=z_dim,out_channels=h_dim*8,kernel_size=4,stride=1,padding=0)
        self.bn0 = nn.BatchNorm2d(h_dim*8)
        self.convt1 = nn.ConvTranspose2d(in_channels=h_dim*8,out_channels=h_dim*4,kernel_size=4,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(h_dim*4)
        self.convt2 = nn.ConvTranspose2d(in_channels=h_dim*4,out_channels=h_dim*2,kernel_size=4,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(h_dim*2)
        self.convt3 = nn.ConvTranspose2d(in_channels=h_dim*2,out_channels=h_dim*1,kernel_size=4,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(h_dim*1)
        self.convt4 = nn.ConvTranspose2d(in_channels=h_dim*1,out_channels=channels,kernel_size=4,stride=2,padding=1)

    def forward(self,noise):
        x = self.bn0(self.convt0(noise))
        x = nn.ReLU(inplace=True)(x)
        x = self.bn1(self.convt1(x))
        x = nn.ReLU(inplace=True)(x)
        x = self.bn2(self.convt2(x))
        x = nn.ReLU(inplace=True)(x)
        x = self.bn3(self.convt3(x))
        x = nn.ReLU(inplace=True)(x)
        x = self.convt4(x)
        x = nn.Tanh()(x)
        return x


class DCGAN_Discriminator(nn.Module):
    '''
    h_dim: first hidden layer dimension
    channels:  channels: data channels (RGB:3, GRAYSCALE: 1)

    '''
    def __init__(self, h_dim=128, channels =1):
        super(DCGAN_Discriminator, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(h_dim)
        self.conv1 = nn.Conv2d(in_channels=h_dim, out_channels=h_dim*2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(h_dim*2)
        self.conv2 = nn.Conv2d(in_channels=h_dim*2, out_channels=h_dim*4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(h_dim*4)
        self.conv3 = nn.Conv2d(in_channels=h_dim*4, out_channels=h_dim*8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(h_dim*8)
        self.conv4 = nn.Conv2d(in_channels=h_dim*8, out_channels=1, kernel_size=4, stride=2, padding=0)

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.bn1(self.conv1(x))
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.bn2(self.conv2(x))
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.bn3(self.conv3(x))
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = x.view(x.shape[0],-1)
        return x
