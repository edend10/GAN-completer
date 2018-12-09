import argparse
import os
import numpy as np
from torch.autograd import Variable
import torch
import helper
from models import Generator, Discriminator
from visdom import Visdom
from utils import is_cuda, create_noise, generate_center_mask, load_model, save_sample_images, log_sample_images, apply_mask

##############
# Eden Dolev #
# COMS 4731  #
##############

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1000, help='number of completion iterations')
parser.add_argument('--use_cpu', type=bool, default=False, help='if testing on cpu')
parser.add_argument('--percep_coeff', type=float, default=0.1, help='perceptual coefficient aka lambda')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--logging', type=bool, default=False, help='log or not')
parser.add_argument('--log_port', type=int, default=8080, help='visdom log panel port')
parser.add_argument('--blend', type=bool, default=False, help='blend after completion?')
parser.add_argument('--num_batches', type=int, default=10, help='number of batches to evaluate')
opt = parser.parse_args()
print(opt)

# ----------
#  Cuda or cpu
# ----------
cuda = is_cuda(opt.use_cpu)

# Logging init
if opt.logging:
    print("Init logging...")
    d_eval_logger = helper.get_logger(opt.log_port, 'd_eval')
    viz_image_logger = Visdom(port=opt.log_port, env="images")

# Loss function
criteria = torch.nn.BCELoss()

# Load generator and discriminator
generator = Generator(opt.batch_size, opt.latent_dim, opt.channels)
discriminator = Discriminator(opt.batch_size, opt.channels)
for model, model_name in [(generator, 'g'), (discriminator, 'd')]:
    load_model(cuda, model, model_name)


if cuda:
    generator.cuda()
    discriminator.cuda()
    criteria.cuda()

dataloader = helper.load_dataset(opt.dataset, opt.img_size, opt.batch_size)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

num_batches = min(opt.num_batches, len(dataloader))

# ----------
#  Completion
# ----------
masked_imgs = None
generated_fills_for_blend = None
gen_imgs = None
completed_imgs = None
eval_valid = Variable(Tensor(np.ones([opt.batch_size, 1, 1, 1])), requires_grad=False)
avg_d_eval = 0
for i, (imgs, _) in enumerate(dataloader):

    if i == num_batches:
        break

    imgs = imgs.type(Tensor)
    save_sample_images(imgs, 'originals', i)

    z = create_noise(cuda, imgs.shape[0], opt.latent_dim)
    optimizer = torch.optim.Adam([z], lr=opt.lr, betas=(opt.b1, opt.b2))

    img_mask = generate_center_mask(Tensor, opt.img_size, opt.channels, 0.3)
    fill_mask = generate_center_mask(Tensor, opt.img_size, opt.channels, 0.25)
    masked_imgs = apply_mask(Tensor, imgs, img_mask)

    save_sample_images(masked_imgs, 'masked', i)

    for j in range(opt.num_iters):
        if z.grad is not None:
            z.grad.data.zero_()
        discriminator.zero_grad()
        generator.zero_grad()

        gen_imgs = generator(z)

        masked_gen_imgs = apply_mask(Tensor, gen_imgs, img_mask)

        generated_fills_for_blend = apply_mask(Tensor, gen_imgs, 1 - fill_mask)
        generated_fills = apply_mask(Tensor, gen_imgs, 1 - img_mask)
        completed_imgs = generated_fills + masked_imgs

        contextual_loss = torch.norm(torch.abs(masked_gen_imgs - masked_imgs), p=1)

        d_output = discriminator(gen_imgs)

        valid = Variable(Tensor(np.random.uniform(0.8, 1.2, (imgs.shape[0], 1, 1, 1))), requires_grad=False)
        perceptual_loss = criteria(d_output, valid)

        completion_loss = contextual_loss + opt.percep_coeff * perceptual_loss

        completion_loss.backward()
        optimizer.step()

        print("[Batch %d/%d] [Iter %d/%d] [Completion loss: %f]" % (i, num_batches, j, opt.num_iters,
                                                                         completion_loss.item()))

    save_sample_images(gen_imgs, 'generated', i)
    save_sample_images(completed_imgs, 'completed', i)
    if opt.logging:
        log_sample_images(viz_image_logger, completed_imgs, i)

    # ----------
    #  Blending
    # ----------
    blended_batch = helper.blend_batch(masked_imgs, generated_fills_for_blend, Tensor)
    blended_batch_sample = blended_batch[:25]
    if opt.logging:
        log_sample_images(blended_batch_sample, i)
    save_sample_images(viz_image_logger, blended_batch_sample, 'blended', i)

    # ----------
    #  Evaluation
    # ----------
    discriminator.zero_grad()
    completed_d_output = discriminator(blended_batch)
    d_eval = criteria(completed_d_output, eval_valid)
    print("---> [Batch %d/%d] [eval: %f]" % (i, num_batches, float(d_eval)))
    avg_d_eval += float(d_eval)
    if opt.logging:
        d_eval_logger.log(i, float(d_eval))

avg_d_eval /= opt.num_batches
print("Avg eval: %f" % avg_d_eval)






