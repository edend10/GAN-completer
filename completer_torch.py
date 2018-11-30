import argparse
import os
import numpy as np
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torch
import helper
from models import Generator, Discriminator
from visdom import Visdom

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--logging', type=bool, default=False, help='log or not')
parser.add_argument('--log_port', type=int, default=8080, help='visdom log panel port')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

if cuda:
    print("Using Cuda!")
else:
    print("No Cuda :(")


def create_noise(batch_size, latent_dim):
    return Variable(Tensor(batch_size, latent_dim).normal_().view(-1, latent_dim, 1, 1))


# Logging
if opt.logging:
    d_real_loss_logger = helper.get_logger(opt.log_port, 'd_loss_real')
    d_fake_loss_logger = helper.get_logger(opt.log_port, 'd_loss_fake')
    d_total_loss_logger = helper.get_logger(opt.log_port, 'd_loss_total')
    g_loss_logger = helper.get_logger(opt.log_port, 'g_loss')
    viz_image_logger = Visdom(port=opt.log_port, env="images")

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt.batch_size, opt.latent_dim, opt.channels)
discriminator = Discriminator(opt.batch_size, opt.channels)


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

dataloader = helper.load_dataset(opt.dataset, opt.img_size, opt.batch_size)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    avg_g_loss = 0
    avg_d_fake_loss = 0
    avg_d_real_loss = 0
    avg_d_total_loss = 0
    epoch_batches = 1
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths with smoothing
        valid = Variable(Tensor(np.random.uniform(0.8, 1.2, (imgs.shape[0], 1))), requires_grad=False)
        fake = Variable(Tensor(np.random.uniform(0.0, 0.15, (imgs.shape[0], 1))), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # D real loss
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        # D fake loss
        z = create_noise(imgs.shape[0], opt.latent_dim)
        gen_imgs = generator(z)

        fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
        d_loss = real_loss + fake_loss

        discriminator.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise as generator input
        z = create_noise(imgs.shape[0], opt.latent_dim)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        discriminator.zero_grad()
        generator.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))
        # for logging purposes
        avg_d_real_loss += float(real_loss)
        avg_d_fake_loss += float(fake_loss)
        avg_d_total_loss += float(d_loss)
        avg_g_loss += float(g_loss)
        epoch_batches += 1

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_images = gen_imgs.data[:25]
            save_image(sample_images, 'images/%d.png' % batches_done, nrow=5, normalize=True)
            if opt.logging:
                sample_grid = make_grid(sample_images, nrow=5, normalize=True, scale_each=False, padding=2, pad_value=0)
                viz_image_logger.image(sample_grid, opts=dict(title='%s_b_%d' % (opt.dataset, batches_done)))

    # log epoch losses
    avg_d_real_loss /= epoch_batches
    avg_d_fake_loss /= epoch_batches
    avg_d_total_loss /= epoch_batches
    avg_g_loss /= epoch_batches
    if opt.logging:
        d_real_loss_logger.log(epoch, avg_d_real_loss)
        d_fake_loss_logger.log(epoch, avg_d_fake_loss)
        d_total_loss_logger.log(epoch, avg_d_total_loss)
        g_loss_logger.log(epoch, avg_g_loss)

    # checkpoint
    torch.save(generator.state_dict(), "g_model")
    torch.save(discriminator.state_dict(), "d_model")
