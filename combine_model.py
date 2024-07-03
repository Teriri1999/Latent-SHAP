import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AlexNetMNIST(nn.Module):
    def __init__(self):
        super(AlexNetMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, g_ema, alexnet, transform):
        super(CombinedModel, self).__init__()
        self.g_ema = g_ema
        self.alexnet = alexnet
        self.transform = transform
        self.normalize = transforms.Normalize((0.5,), (0.5,))

    def forward(self, z, input_is_latent=False):
        if input_is_latent:
            img_gen, _ = self.g_ema([z], input_is_latent=True, noise=None, randomize_noise=False)
        else:
            img_gen, _ = self.g_ema([z], input_is_latent=False, noise=None, randomize_noise=False)

        # vis = make_image(img_gen)
        # plt.imshow(vis[0])
        img_gen = (img_gen + 1) / 2
        img_gen = torch.mean(img_gen, dim=1, keepdim=True)
        img_gen = nn.functional.interpolate(img_gen, size=(224, 224), mode='bilinear', align_corners=False)
        img_gen = self.normalize(img_gen)

        output = self.alexnet(img_gen)

        return output
