"""Hybrid ResNet50 + DeiT-Tiny model used by training and the web app."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn
from torchvision import models


CLASS_NAMES = ("Pothole", "Asphalt", "Crack")


class HybridRoadDamageModel(nn.Module):
    """Fuse CNN local features and transformer global features."""

    def __init__(
        self,
        num_classes: int = len(CLASS_NAMES),
        pretrained: bool = True,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        resnet_weights = (
            models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        resnet = models.resnet50(weights=resnet_weights)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.transformer_backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        cnn_features = 2048
        transformer_features = self.transformer_backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(cnn_features + transformer_features),
            nn.Linear(cnn_features + transformer_features, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )
        if freeze_backbones:
            for parameter in self.cnn_backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.transformer_backbone.parameters():
                parameter.requires_grad = False

    def forward_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_map = self.cnn_backbone(images)
        cnn_vector = self.cnn_pool(cnn_map).flatten(1)
        transformer_vector = self.transformer_backbone(images)
        return cnn_map, torch.cat((cnn_vector, transformer_vector), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, fused_features = self.forward_features(images)
        return self.classifier(fused_features)


def build_hybrid_model(
    checkpoint_path: Path | None = None,
    pretrained: bool = True,
    freeze_backbones: bool = False,
    device: torch.device | None = None,
) -> HybridRoadDamageModel:
    """Build the hybrid model and optionally load a saved checkpoint."""
    model = HybridRoadDamageModel(
        pretrained=pretrained,
        freeze_backbones=freeze_backbones,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    return model
