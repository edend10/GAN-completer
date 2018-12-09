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
parser.add_argument('--sample_interval', type=int, default=100, help='interval between image sampling')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--logging', type=bool, default=False, help='log or not')
parser.add_argument('--log_port', type=int, default=8080, help='visdom log panel port')
parser.add_argument('--blend', type=bool, default=False, help='blend after completion?')
parser.add_argument('--num_batches', type=int, default=1, help='number of batches to evaluate')
parser.add_argument('--mask_type', type=int, default=1, help='mask type')

opt = parser.parse_args()
print(opt)

cuda = is_cuda(opt.use_cpu)

# Logging
if opt.logging:
    print("Init logging...")
    contextual_loss_logger = helper.get_logger(opt.log_port, 'contextual_loss')
    perceptual_loss_logger = helper.get_logger(opt.log_port, 'perceptual_loss')
    completion_loss_logger = helper.get_logger(opt.log_port, 'completion_loss')
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
for i, (imgs, _) in enumerate(dataloader):

    if i == num_batches:
        break

    imgs = imgs.type(Tensor)
    save_sample_images(imgs, 'originals', i)

    # initial input for generator. This is what we want to optimize
    z = create_noise(cuda, imgs.shape[0], opt.latent_dim)
    optimizer = torch.optim.Adam([z], lr=opt.lr, betas=(opt.b1, opt.b2))

    img_mask = generate_center_mask(Tensor, opt.img_size, opt.channels, 0.3)
    # we also want a slightly bigger version of the mask for alpha blending later
    fill_mask = generate_center_mask(Tensor, opt.img_size, opt.channels, 0.25)
    masked_imgs = apply_mask(Tensor, imgs, img_mask)

    save_sample_images(masked_imgs, 'masked', i)

    avg_contextual_loss = 0
    avg_perceptual_loss = 0
    avg_completion_loss = 0

    # iterate n times over the same batch and optimize to find the best z input vector
    # that will produce the best completed result
    for j in range(opt.num_iters):
        if z.grad is not None:
            z.grad.data.zero_()
        discriminator.zero_grad()
        generator.zero_grad()

        gen_imgs = generator(z)

        # apply mask to generated fake images
        masked_gen_imgs = apply_mask(Tensor, gen_imgs, img_mask)

        # crop out a slightly bigger fill from the fake image than the missing part of the original image
        # for later blending purposes
        generated_fills_for_blend = apply_mask(Tensor, gen_imgs, 1 - fill_mask)

        # save/log a sample of completed images (masked original + fill)
        generated_fills = apply_mask(Tensor, gen_imgs, 1 - img_mask)
        completed_imgs = generated_fills + masked_imgs
        if j % opt.sample_interval == 0:
            save_sample_images(gen_imgs, 'generated', [i, j])
            save_sample_images(completed_imgs, 'completed', [i, j])
            if opt.logging:
                log_sample_images(viz_image_logger, completed_imgs, [i, j])

        # calculate the contextual loss.
        # the pixels we have color for in the original image minus the same pixels in the generated image.
        # this is how we measure the similarity between what we generated and the original image
        contextual_loss = torch.norm(torch.abs(masked_gen_imgs - masked_imgs), p=1)

        # calculate the perceptual loss.
        # similar to how we originally trained the generator.
        # this is how we measure the ability of the generator image to fool the discriminator.
        # it keeps the generated images "realistic"
        d_output = discriminator(gen_imgs)
        valid = Variable(Tensor(np.random.uniform(0.8, 1.2, (imgs.shape[0], 1, 1, 1))), requires_grad=False)
        perceptual_loss = criteria(d_output, valid)

        completion_loss = contextual_loss + opt.percep_coeff * perceptual_loss

        if opt.logging:
            avg_contextual_loss += float(contextual_loss)
            avg_perceptual_loss += float(perceptual_loss)
            avg_completion_loss += float(completion_loss)
            if (j+1) % opt.sample_interval == 0:
                avg_contextual_loss /= opt.sample_interval
                avg_perceptual_loss /= opt.sample_interval
                avg_completion_loss /= opt.sample_interval
                contextual_loss_logger.log(j, avg_contextual_loss)
                perceptual_loss_logger.log(j, avg_perceptual_loss)
                completion_loss_logger.log(j, avg_completion_loss)
                avg_contextual_loss = 0
                avg_perceptual_loss = 0
                avg_completion_loss = 0

        completion_loss.backward()
        optimizer.step()

        print("[Batch %d/%d] [Iter %d/%d] [Completion loss: %f]" % (i, num_batches, j, opt.num_iters,
                                                                         completion_loss.item()))

    # ----------
    #  Blending
    # ----------
    if (opt.blend):
        # apply alpha blending to the images.
        # for alpha blending we use a "fill" crop from the generated fake images that is
        # slightly larger than the missing part of the original image
        if opt.logging:
            log_sample_images(viz_image_logger, masked_imgs, "masked")
            log_sample_images(viz_image_logger, generated_fills_for_blend, "fills")
        blended_batch = helper.blend_batch(masked_imgs[:25], generated_fills_for_blend[:25], Tensor)
        if opt.logging:
            log_sample_images(viz_image_logger, blended_batch, i)
        save_sample_images(blended_batch, 'blended', i)

