import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

class CustomEfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, pretrained=False):
        """
        Custom EfficientNet-B3 for binary classification with additional flexibility.

        Args:
            num_classes (int): Number of output classes. Default is 1 for binary classification.
            dropout_rate (float): Dropout rate for the classifier layer. Default is 0.3.
            pretrained (bool): Whether to load pretrained weights. Default is False.
        """
        super(CustomEfficientNetB3Classifier, self).__init__()
        self.efficient_net = efficientnet_b3(pretrained=pretrained)

        # Extract features
        self.features = self.efficient_net.features

        # Modify the classifier layer
        in_features = self.efficient_net.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
       
        self.activation = None

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output logits or probabilities.
        """
        x = self.features(x)  # Pass through feature extractor
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten for the classifier
        x = self.classifier(x)  # Pass through the classifier
        return x

    def freeze_base(self):
        """
        Freezes the base feature extractor for transfer learning.
        """
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        """
        Unfreezes the base feature extractor for fine-tuning.
        """
        for param in self.features.parameters():
            param.requires_grad = True


def CustomEfficientNetB3(num_classes=1, dropout_rate=0.3, pretrained=False):
    """
    Factory function to create a Custom EfficientNet-B3 model.

    Args:
        num_classes (int): Number of output classes. Default is 1 for binary classification.
        dropout_rate (float): Dropout rate for the classifier layer. Default is 0.3.
        pretrained (bool): Whether to load pretrained weights. Default is False.

    Returns:
        nn.Module: Custom EfficientNet-B3 model instance.
    """
    return CustomEfficientNetB3Classifier(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=pretrained)
