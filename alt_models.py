import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, image_size, ngf, latent_dimension, num_channels):
        super(Generator, self).__init__()

        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dimension, ngf * 8 * self.init_size**2))

        self.ngf = ngf

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, num_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.ngf * 8, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, image_size, ndf, num_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(num_channels, ndf, bn=False),
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            *discriminator_block(ndf * 4, ndf * 8),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(ndf * 8 * ds_size**2, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
