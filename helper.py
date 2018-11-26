import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchnet.logger import VisdomPlotLogger


def load_dataset(dataset_name, image_size, batch_size):
    path = 'data/' + dataset_name
    normal_mean = (0.1, 0.1, 0.1)
    normal_std = (0.1, 0.1, 0.1)
    os.makedirs(path, exist_ok=True)
    if dataset_name == 'cifar10':
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(normal_mean, normal_std)
                               ])),
            batch_size=batch_size, shuffle=True)

    elif dataset_name == 'mnist':
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(normal_mean, normal_std)
                           ])),
            batch_size=batch_size, shuffle=True)

    elif dataset_name == 'stl10':
        dataloader = torch.utils.data.DataLoader(
            datasets.STL10(path, 'train', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(normal_mean, normal_std)
                           ])),
            batch_size=batch_size, shuffle=True)
    elif dataset_name == 'celeba':
        dataloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path,
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(128),
                                     transforms.Resize(image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(normal_mean, normal_std)
                                 ])),
            batch_size=batch_size, shuffle=True)

    return dataloader


def get_logger(port, name):
    return VisdomPlotLogger('line', port=port, opts={'title': '%s' % name})
