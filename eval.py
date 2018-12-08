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
parser.add_argument('--num_iters', type=int, default=1000, help='number of completion iterations')
parser.add_argument('--use_cpu', type=bool, default=False, help='if testing on cpu')
parser.add_argument('--percep_coeff', type=float, default=0.1, help='perceptual coefficient aka lambda')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--logging', type=bool, default=False, help='log or not')
parser.add_argument('--log_port', type=int, default=8080, help='visdom log panel port')
parser.add_argument('--debug', type=bool, default=False, help='debug mode')
parser.add_argument('--blend', type=bool, default=False, help='blend after completion?')
parser.add_argument('--mask', type=str, default='center', help='center/random mask')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() and not opt.use_cpu else False

if cuda:
    print("Using Cuda!")
else:
    print("No Cuda :(")


def create_noise(batch_size, latent_dim):
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    # return Variable(Tensor(batch_size, latent_dim).normal_().view(-1, latent_dim, 1, 1))
    return torch.rand(size=[batch_size, latent_dim, 1, 1], dtype=torch.float32, requires_grad=True, device=device)


def generate_center_mask(img_size, num_channels, center_scale=0.3):
    img_shape = (num_channels, img_size, img_size)

    mask = torch.ones(size=img_shape).type(Tensor)
    low = int(img_size * center_scale)
    high = int(img_size * (1 - center_scale))
    mask[:, low:high, low:high] = 0
    return mask


def load_model(model, model_name):
    print("Loading model: `%s`" % model_name)
    model_path = 'models/%s_model' % model_name
    if not cuda:
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)


def save_sample_images(imgs, sample_dir, img_ids):
    if type(img_ids) == list:
        str_id_path = '_'.join([str(i) for i in img_ids])
    else:
        str_id_path = str(img_ids)

    if opt.debug:
        print("Saving images: `%s`" % sample_dir)
    sample_images = imgs.data[:25]
    save_image(sample_images, 'images/completion/%s/%s.png' % (sample_dir, str_id_path), nrow=5, normalize=True)


def log_sample_images(imgs, img_ids):
    if type(img_ids) == list:
        str_id_path = '_'.join([str(i) for i in img_ids])
    else:
        str_id_path = str(img_ids)

    sample_images = imgs.data[:25]
    sample_grid = make_grid(sample_images, nrow=5, normalize=True, scale_each=False, padding=2, pad_value=0)
    viz_image_logger.image(sample_grid, opts=dict(title=str_id_path))


def apply_mask(img, mask):
    return torch.mul(img+1, mask).type(Tensor) - 1


# Logging
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
    load_model(model, model_name)


if cuda:
    generator.cuda()
    discriminator.cuda()
    criteria.cuda()

dataloader = helper.load_dataset(opt.dataset, opt.img_size, opt.batch_size)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Completion
# ----------
masked_imgs = None
generated_fills_for_blend = None
gen_imgs = None
completed_imgs = None
eval_valid = Variable(Tensor(np.ones([opt.batch_size, 1, 1, 1])), requires_grad=False)
for i, (imgs, _) in enumerate(dataloader):

    if i == 10:
        break

    imgs = imgs.type(Tensor)
    save_sample_images(imgs, 'originals', i)

    z = create_noise(imgs.shape[0], opt.latent_dim)
    optimizer = torch.optim.Adam([z], lr=opt.lr, betas=(opt.b1, opt.b2))

    img_mask = generate_center_mask(opt.img_size, opt.channels, 0.3)
    fill_mask = generate_center_mask(opt.img_size, opt.channels, 0.25)
    masked_imgs = apply_mask(imgs, img_mask)

    save_sample_images(masked_imgs, 'masked', i)

    for j in range(opt.num_iters):
        if z.grad is not None:
            z.grad.data.zero_()
        discriminator.zero_grad()
        generator.zero_grad()

        gen_imgs = generator(z)

        masked_gen_imgs = apply_mask(gen_imgs, img_mask)

        generated_fills_for_blend = apply_mask(gen_imgs, 1 - fill_mask)
        generated_fills = apply_mask(gen_imgs, 1 - img_mask)
        completed_imgs = generated_fills + masked_imgs

        contextual_loss = torch.norm(torch.abs(masked_gen_imgs - masked_imgs), p=1)

        d_output = discriminator(gen_imgs)

        valid = Variable(Tensor(np.random.uniform(0.8, 1.2, (imgs.shape[0], 1, 1, 1))), requires_grad=False)
        perceptual_loss = criteria(d_output, valid)

        completion_loss = contextual_loss + opt.percep_coeff * perceptual_loss

        completion_loss.backward()
        optimizer.step()

        print("[Batch %d/%d] [Iter %d/%d] [Completion loss: %f]" % (i, len(dataloader), j, opt.num_iters,
                                                                         completion_loss.item()))

    save_sample_images(gen_imgs, 'generated', i)
    save_sample_images(completed_imgs, 'completed', i)
    if opt.logging:
        log_sample_images(completed_imgs, i)

    # ----------
    #  Blending
    # ----------
    if opt.logging:
        log_sample_images(masked_imgs, "masked")
        log_sample_images(generated_fills_for_blend, "fills")
    blended_batch = helper.blend_batch(masked_imgs, generated_fills_for_blend, Tensor)
    blended_batch_sample = blended_batch[:25]
    if opt.logging:
        log_sample_images(blended_batch_sample, i)
    save_sample_images(blended_batch_sample, 'blended', i)

    # ----------
    #  Evaluation
    # ----------
    completed_d_output = discriminator(blended_batch)
    d_eval = criteria(completed_d_output, eval_valid)
    print("---> [Batch %d/%d] [eval: %f]" % (i, len(dataloader), d_eval.item()))
    if opt.logging:
        d_eval_logger.log(i, float(d_eval))






