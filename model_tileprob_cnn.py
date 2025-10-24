"""
A lightweight PyTorch CNN that consumes tile-prob matrices (C x H x W) and outputs class logits.
It uses convolutional layers and an AdaptiveAvgPool2d to handle variable HxW.

"""
import torch
import torch.nn as nn


class TileProbCNN(nn.Module):
    def __init__(self, in_channels=11, num_classes=200, hidden=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden*2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden*2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    m = TileProbCNN(in_channels=11, num_classes=100)
    import torch
    dummy = torch.randn(4,11,6,6)
    out = m(dummy)
    print('out', out.shape)
