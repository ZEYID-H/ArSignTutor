import sys
from pathlib import Path
import torch.nn as nn
from torchvision import models

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config


class ArSignModel(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        feats = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(feats, num_classes))

    def forward(self, x):
        return self.model(x)
