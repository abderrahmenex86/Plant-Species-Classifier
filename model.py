import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class PlantClassifier(nn.Module):
    def __init__(self, num_classes=1081):
        super().__init__()
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        in_features = self.model.classifier[3].in_features

        self.model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
