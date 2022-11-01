import timm
import torch
import torch.nn as nn
import torchvision
from config import cfg

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def select_model(model_name):
    if model_name == "efficientnet":
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=3)
    elif model_name == "resnet18":
        model = timm.create_model('resnet18d', pretrained=True, num_classes=3)
    elif model_name == "resnet34":
        model = timm.create_model('resnet34d', pretrained=True, num_classes=3)
    elif model_name == "resnet50":
        model = timm.create_model('resnet50d', pretrained=True, num_classes=3)
    elif model_name == "resnet101":
        model = timm.create_model('resnet101d', pretrained=True, num_classes=3)
    elif model_name == 'DenseNet121':
        model = DenseNet121(cfg.out_size)
        
    return model
