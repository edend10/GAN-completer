import helper
import argparse
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')

opt = parser.parse_args()

dataloader = helper.load_dataset(opt.dataset, opt.img_size, opt.batch_size)

for i, (imgs, _) in enumerate(dataloader):
    if i == 1:
        break
    save_image(imgs.data[:25], 'images/dataset_test.png', nrow=5, normalize=True)
