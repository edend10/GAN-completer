import helper
import argparse
from torchvision.utils import save_image, make_grid
from visdom import Visdom

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')

opt = parser.parse_args()

dataloader = helper.load_dataset(opt.dataset, opt.img_size, opt.batch_size)
viz = Visdom(port=8080, env="main")
viz2 = Visdom(port=8080, env="images")
for i, (imgs, _) in enumerate(dataloader):
    if i == 1:
        break
    grid = make_grid(imgs.data[:25], nrow=5, padding=2, pad_value=0,
                     normalize=True, scale_each=False)
    grid2 = make_grid(imgs.data[25:50], nrow=5, padding=2, pad_value=0,
                     normalize=True, scale_each=False)
    viz.image(grid, opts=dict(title='Random!', caption='How random.'))
    viz2.image(grid2, opts=dict(title='Random2!', caption='How random2.'))

    save_image(imgs.data[:25], 'images/dataset_test.png', nrow=5, normalize=True)
