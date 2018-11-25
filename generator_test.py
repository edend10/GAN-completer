import torch
import numpy as np
import argparse
from models import Generator
from torch.autograd import Variable
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--batch_size', type=int, default=9, help='size of the batches')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--cpu', type=bool, default=True, help='if testing on cpu')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    print("Using Cuda!")
else:
    print("No Cuda :(")

generator = Generator(opt.img_size, opt.latent_dim, opt.channels)
if opt.cpu:
    model = torch.load('g_model', map_location='cpu')
else:
    model = torch.load('g_model')

generator.load_state_dict(model)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

gen_imgs = generator(z)

save_image(gen_imgs.data, 'images/output.png', nrow=3, normalize=True)
