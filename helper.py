import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from visualdl import LogWriter


def load_dataset(dataset_name, image_size, batch_size):
    path = 'data/' + dataset_name
    os.makedirs(path, exist_ok=True)
    if dataset_name == 'cifar10':
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])),
            batch_size = batch_size, shuffle = True)

    elif dataset_name == 'mnist':
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])),
            batch_size=batch_size, shuffle=True)

    elif dataset_name == 'stl10':
        dataloader = torch.utils.data.DataLoader(
            datasets.STL10(path, 'train', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])),
            batch_size=batch_size, shuffle=True)

    return dataloader


def get_loggers(log_dir, sync_cycle=100):
    logger = LogWriter(log_dir, sync_cycle=sync_cycle)

    with logger.mode("train"):
        # create a scalar component called 'scalars/'
        d_loss_log_scalar = logger.scalar("scalars/d_loss_log_scalar")
        g_loss_log_scalar = logger.scalar("scalars/g_loss_log_scalar")

    return d_loss_log_scalar, g_loss_log_scalar
