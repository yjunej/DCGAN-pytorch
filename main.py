from utils import *
from models import DCGAN_Generator, DCGAN_Discriminator
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Deep Convolutional Generative Adversirial Networks pytorch Implementation based on the original research paper")

parser.add_argument('-lr',type=float, default=0.0002, help='Train Learning Rate')
parser.add_argument('-dataset',type=str, default='MNIST', help='Dataset: MNIST, CelebA')
parser.add_argument('-batch_size',type=int, default=128, help='Train Batch size')
parser.add_argument('-hidden_dim',type=int, default=128, help='Train hidden channel dimension ')
parser.add_argument('-z_dim',type=int, default=100, help='Noise Space Z dimention')
parser.add_argument('-beta_1',type=float, default=0.5, help='Beta 1 for Adam Optimizer')
parser.add_argument('-beta_2',type=float, default=0.999, help='Beta 2 for Adam Optimizer')
parser.add_argument('-epochs',type=int, default=30, help='Numbers of Train Epochs')
parser.add_argument('-optimizer',type=str, default='Adam', help='Select Optimizer: Adam, SGD, if SGD selected, beta parameters would be deactivated')
parser.add_argument('-visual_batch_step', type=int, default=60, help='Visualizing batch interval for fixed z')
parser.add_argument('-denormalize_img',type=bool, default=True, help='image_tensor * 0.5 + 0.5 for clean visualization')
args = parser.parse_args()

criterion = nn.BCEWithLogitsLoss()
learning_rate = args.lr 
beta = (args.beta_1, args.beta_2)
z_dim = args.z_dim
h_dim = args.hidden_dim
epochs = args.epochs
batch_size = args.batch_size

print('Downloading Dataset'.center(60,'='))
if args.dataset == 'MNIST':
    mnist_transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize((64,64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((.5),(.5))
    ])

    mnist = torchvision.datasets.MNIST('./data',download=True, transform=mnist_transform)
    data_loader = torch.utils.data.DataLoader(mnist,batch_size=batch_size,shuffle=True,drop_last=True)
    channels = 1

elif args.dataset == 'CelebA':
    celeba_transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize((64,64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((.5),(.5),(.5))
    ])
    celeba = torchvision.datasets.CelebA('./data',download=True, transform=celeba_transform)
    data_loader = torch.utils.data.DataLoader(celeba,batch_size=batch_size,shuffle=True,drop_last=True)
    channels = 3

else:
    try:
        custom_transform = torchvision.transfroms.Compose([
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((.5),(.5),(.5))
        ])
        channels = 3
    except:
        custom_transform = torchvision.transfroms.Compose([
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((.5),(.5))
        ])
        channels = 1
    dataset = torchvision.dataset.ImageFolder(root=args.dataset, transform=custom_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, drop_last=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

generator = DCGAN_Generator(z_dim=z_dim, h_dim=h_dim, channels=channels).to(device)
discriminator = DCGAN_Discriminator(h_dim=h_dim, channels=channels).to(device)

if args.optimizer == 'Adam':
    gen_optimizer = torch.optim.Adam(generator.parameters(),lr=learning_rate,betas=beta)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(),lr=learning_rate,betas=beta)

else:
    gen_optimizer = torch.optim.SGD(generator.parameters(),lr=learning_rate)
    dis_optimizer = torch.optim.SGD(discriminator.parameters(),lr=learning_rate)


generator = generator.apply(weight_initialize)
discriminator = discriminator.apply(weight_initialize)


gen_loss_list = []
dis_loss_list = []


def train():
    print('Start Training'.center(60,'='))
    make_result_dir()
    
    batch_count = 0
    example_z = noise_maker(batch_size=batch_size,z_dim=z_dim,device=device)
    for i in tqdm(range(epochs)):

            step_per_epoch = 0
            mean_gen_loss = 0
            mean_dis_loss = 0

            for x_real, _ in data_loader:
                step_per_epoch += 1
                
                x_real = x_real.to(device)

                dis_optimizer.zero_grad()
                dis_pred_real = discriminator(x_real)
                dis_loss_real = criterion(dis_pred_real, torch.ones_like(dis_pred_real))

                x_gen = generator(noise_maker(batch_size, z_dim, device))
                x_gen = x_gen.detach()
                dis_pred_gen = discriminator(x_gen)
                dis_loss_gen = criterion(dis_pred_gen, torch.zeros_like(dis_pred_gen))

                dis_loss = (dis_loss_real + dis_loss_gen) / 2
                mean_dis_loss += dis_loss
                dis_loss.backward()
                dis_optimizer.step()

                gen_optimizer.zero_grad()
                x_gen_ = generator(noise_maker(batch_size, z_dim, device))
                dis_pred_gen = discriminator(x_gen_)

                gen_loss = criterion(dis_pred_gen, torch.ones_like(dis_pred_gen))
                mean_gen_loss += gen_loss
                gen_loss.backward()
                gen_optimizer.step()


                if batch_count%args.visual_batch_step == 0:
                    gen_example_z = generator(example_z)
                    visualization_grid(gen_example_z,size=(1,64,64),num=36,step=batch_count,epochs=i)
            
                batch_count += 1

            mean_gen_loss /= step_per_epoch
            mean_dis_loss /= step_per_epoch
            gen_loss_list.append(mean_gen_loss)
            dis_loss_list.append(mean_dis_loss)

train()
print('Training Done, Check result folder'.center(60,'='))
