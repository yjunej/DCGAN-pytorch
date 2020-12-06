from utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr',type=float, default=0.0002)
parser.add_argument('--batch_size',type=int,required=False, default=128)
parser.add_argument('--hidden_dim',type=int,required=False, default=128)


mnist_transform = torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((.5),(.5))
])

mnist = torchvision.datasets.MNIST('./data',download=True, transform=mnist_transform)
mnist_loader = torch.utils.data.DataLoader(mnist,batch_size=128,shuffle=True,drop_last=True)

criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.0002 #0.001
beta = (0.5, 0.999)
z_dim = 100
epochs = 30

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

generator = DCGAN_Generator().to(device)
discriminator = DCGAN_Discrimanator().to(device)

gen_optimizer = torch.optim.Adam(generator.parameters(),lr=learning_rate,betas=beta)
dis_optimizer = torch.optim.Adam(generator.parameters(),lr=learning_rate,betas=beta)

generator = generator.apply(weight_initialize)
discriminator = discriminator.apply(weight_initialize)

def train():
    # gen_loss = 0
    # disc_loss = 0
    batch_count = 0
    example_z = noise_maker(batch_size=36,z_dim=100,device=device)
    for i in range(epochs):
            for x_real, label in tqdm(mnist_loader):
                batch_count += 1
                x_real = x_real.to(device)

                dis_optimizer.zero_grad()
                dis_pred_real = discriminator(x_real)
                dis_loss_real = criterion(dis_pred_real, torch.ones_like(dis_pred_real))

                x_gen = gen(noise_maker(batch_size, z_dim, device))
                x_gen = x_generated.detach()
                dis_pred_gen = discriminator(x_gen)
                dis_loss_gen = criterion(dis_pred_gen, torch.zeros_like(dis_pred_gen))

                dis_loss = (dis_loss_real + dis_loss_gen) / 2
                dis_loss.backward()
                dis_optimizer.step()

                gen_optimizer.zero_grad()
                x_gen_ = gen(noise_maker(batch_size, z_dim, device))
                dis_pred_gen = discriminator(x_gen_)
                gen_loss = criterion(dis_pred_gen, torch.ones_like(dis_pred_gen))

                gen_loss.bakcward()
                gen_optimizer.step()

                if batch_count%60 == 0:
                    pass


