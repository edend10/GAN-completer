import helper
import cv2
import numpy as np
import torch
from utils import generate_random_mask


def wind(img, window=3):
    img_size = img.shape[1]
    img_copy = np.array(img)
    for i in range(img_size):
        for j in range(img_size):
            convolve(img, img_copy, img_size, i, j, window)
    return img_copy


def convolve(img, img_copy, img_size, i, j, window):
    for k in range(0, window):
        n = i - 1 + k
        for l in range(0, window):
            m = j - 1 + l
            if 0 <= n < img_size and 0 <= m < img_size:
                if (img[:, n, m] == 0).all():
                    img_copy[:, i, j] = 0
                    return


def generate_mask(img_size, num_channels, center_scale):
    img_shape = (img_size, img_size, num_channels)

    mask = np.ones(img_shape)
    low = int(img_size * center_scale)
    high = int(img_size * (1 - center_scale))
    mask[low:high, low:high, :] = 0.0
    return mask


def center():
    img = cv2.imread("blend_test/masked/orig.png")
    fill = cv2.imread("blend_test/generated/0_900.png")

    img_mask = generate_mask(64, 3, 0.3)
    fill_mask = generate_mask(64, 3, 0.25)
    fill = np.multiply((1-fill_mask), fill)
    img = np.multiply(img_mask, img)

    naive_completion = img + fill
    cv2.imwrite("blend_test/blended/naive_completion.png", naive_completion)


    img = np.einsum('ijk->kij', img)
    fill = np.einsum('ijk->kij', fill)
    img = torch.FloatTensor(img)
    fill = torch.FloatTensor(fill)
    # naive_completion = np.einsum('ijk->kij', naive_completion)
    #
    #
    # blended = helper.poisson_blend_img(naive_completion, fill)

    blended = helper.alpha_blend_img(img, fill, torch.FloatTensor)
    blended = helper.to_np(blended)
    cv2.imwrite("blend_test/blended/blended_completion.png", blended)


def random():
    img = cv2.imread("blend_test/masked/orig.png")
    generated = cv2.imread("blend_test/generated/0_900.png")

    img = np.einsum('ijk->kij', img)
    generated = np.einsum('ijk->kij', generated)
    img = torch.FloatTensor(img)
    generated = torch.FloatTensor(generated)

    img_mask = generate_random_mask(torch.FloatTensor, 64, 3, 0.1)
    enlarged_mask = wind(img_mask)
    masked_img = np.multiply(img_mask, img)
    fill = np.multiply((1-img_mask), generated)
    enlarged_fill = np.multiply((1-enlarged_mask), generated)

    naive_completion = masked_img + fill
    naive_completion = np.einsum('kij->ijk', naive_completion)
    cv2.imwrite("blend_test/blended/naive_completion.png", naive_completion)

    blended = helper.alpha_blend_img(masked_img, enlarged_fill, torch.FloatTensor)
    blended = helper.to_np(blended)

    fill = np.einsum('kij->ijk', fill)
    enlarged_fill = np.einsum('kij->ijk', enlarged_fill)
    cv2.imwrite("blend_test/generated/fill.png", fill)
    cv2.imwrite("blend_test/generated/enlarged_fill.png", enlarged_fill)

    cv2.imwrite("blend_test/blended/blended_completion.png", blended)


random()
