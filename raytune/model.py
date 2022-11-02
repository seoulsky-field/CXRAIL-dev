import torchvision
import torch
import torch.nn as nn


class DenseNet121(nn.Module):
    def __init__(self, num_classes, is_trained=True):
        super().__init__() 
        
        self.net = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
        kernel_count = self.net.classifier.in_features
        self.net.classifier = nn.Sequential(nn.Linear(kernel_count, num_classes), nn.Sigmoid())
        
    def forward(self, inputs):
        """
        Forward the netword with the inputs
        """
        return self.net(inputs)