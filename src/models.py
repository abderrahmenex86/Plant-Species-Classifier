import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class PlantClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone_network = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        classifier_input_features = self.backbone_network.classifier[3].in_features
        self.backbone_network.classifier[2] = nn.Dropout(p=0.5, inplace=True)
        self.backbone_network.classifier[3] = nn.Linear(classifier_input_features, num_classes)

    def forward(self, input_tensor):
        return self.backbone_network(input_tensor)


class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma_value=2.0, reduction_method="mean"):
        super().__init__()
        self.gamma_value = gamma_value
        self.reduction_method = reduction_method

    def forward(self, model_predictions, target_labels):
        log_probabilities = F.log_softmax(model_predictions, dim=-1)
        probabilities = torch.exp(log_probabilities)
        target_one_hot = F.one_hot(target_labels, num_classes=model_predictions.size(-1)).to(model_predictions.dtype)
        focal_weights = torch.pow(1.0 - probabilities, self.gamma_value)
        loss_tensor = -target_one_hot * focal_weights * log_probabilities
        summed_loss = torch.sum(loss_tensor, dim=-1)

        if self.reduction_method == "mean":
            return torch.mean(summed_loss)
        elif self.reduction_method == "sum":
            return torch.sum(summed_loss)
        return summed_loss
