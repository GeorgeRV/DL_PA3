from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    """CNN compacta treinada do zero, usada como baseline da Parte 0(a)."""

    def __init__(self, num_classes: int = 21):
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int = 21, pretrained: bool = True) -> nn.Module:
    model_name = model_name.casefold()
    if model_name in {"cnn", "cnn_scratch", "small_cnn"}:
        return SmallCNN(num_classes=num_classes)

    if model_name in {"resnet18", "resnet-18"}:
        if pretrained:
            try:
                # Pesos ImageNet-1K V1: escolha explícita para maior reprodutibilidade.
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                model = models.resnet18(weights=weights)
            except AttributeError:
                # Compatibilidade com versões antigas do torchvision.
                model = models.resnet18(pretrained=True)
        else:
            try:
                model = models.resnet18(weights=None)
            except TypeError:
                model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Modelo desconhecido: {model_name}. Use 'cnn_scratch' ou 'resnet18'.")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
