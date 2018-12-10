from torchvision.utils import save_image, make_grid
import torch
import numpy as np

##############
# Eden Dolev #
##############


# Create random z vector for generator input
def create_noise(cuda, batch_size, latent_dim):
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    return torch.rand(size=[batch_size, latent_dim, 1, 1], dtype=torch.float32, requires_grad=True, device=device)


# Generate mask over image center
def generate_center_mask(Tensor, img_size, num_channels, center_scale=0.3):
    img_shape = (num_channels, img_size, img_size)

    mask = torch.ones(size=img_shape).type(Tensor)
    low = int(img_size * center_scale)
    high = int(img_size * (1 - center_scale))
    mask[:, low:high, low:high] = 0
    return mask


def generate_random_mask(Tensor, img_size, num_channels, mask_threshold=0.2):
    mask = np.ones([num_channels, img_size, img_size])
    mask[:, np.random.random([img_size, img_size]) < mask_threshold] = 0.0
    return Tensor(mask)


# Load NN model
def load_model(cuda, model, model_name):
    print("Loading model: `%s`" % model_name)
    model_path = 'models/%s_model' % model_name
    if not cuda:
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)


# Save sample image
def save_sample_images(imgs, sample_dir, img_ids):
    if type(img_ids) == list:
        str_id_path = '_'.join([str(i) for i in img_ids])
    else:
        str_id_path = str(img_ids)

    sample_images = imgs.data[:25]
    save_image(sample_images, 'images/completion/%s/%s.png' % (sample_dir, str_id_path), nrow=5, normalize=True)


# Log sample images to Visdom
def log_sample_images(image_logger, imgs, img_ids):
    if type(img_ids) == list:
        str_id_path = '_'.join([str(i) for i in img_ids])
    else:
        str_id_path = str(img_ids)

    sample_images = imgs.data[:25]
    sample_grid = make_grid(sample_images, nrow=5, normalize=True, scale_each=False, padding=2, pad_value=0)
    image_logger.image(sample_grid, opts=dict(title=str_id_path))


# Apply mask to image
def apply_mask(Tensor, img, mask):
    # torch image has -1 as min value so we want to shift it to 0 min before applying the mask
    # after applying the mask we shift it back
    return torch.mul(img + 1, mask).type(Tensor) - 1


def is_cuda(use_cpu):
    cuda = True if torch.cuda.is_available() and not use_cpu else False

    if cuda:
        print("Using Cuda!")
    else:
        print("No Cuda :(")

    return cuda


