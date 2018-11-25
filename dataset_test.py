import helper
from torchvision.utils import save_image


dataloader = helper.load_dataset('celeba', 64, 64)

for i, (imgs, _) in enumerate(dataloader):
    if i == 1:
        break
    save_image(imgs.data[:25], 'images/dataset_test.png', nrow=5, normalize=True)
