import torch

Tensor = torch.FloatTensor

def generate_mask(img_size, num_channels):
    img_shape = (num_channels, img_size, img_size)

    mask = torch.ones(size=img_shape).type(Tensor)
    center_scale = 0.3
    low = int(img_size * center_scale)
    high = int(img_size * (1 - center_scale))
    mask[:, low:high, low:high] = 0.0
    return mask