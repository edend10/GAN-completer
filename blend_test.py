import helper
import cv2
import numpy as np


def generate_mask(img_size, num_channels, center_scale):
    img_shape = (img_size, img_size, num_channels)

    mask = np.ones(img_shape)
    low = int(img_size * center_scale)
    high = int(img_size * (1 - center_scale))
    mask[low:high, low:high, :] = 0.0
    return mask


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
# naive_completion = np.einsum('ijk->kij', naive_completion)
#
#
# blended = helper.poisson_blend_img(naive_completion, fill)

blended = helper.alpha_blend_img(img, fill)
cv2.imwrite("blend_test/blended/blended_completion.png", blended)

