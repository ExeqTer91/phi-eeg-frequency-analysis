"""SimpleCNN with configurable layer widths for φ-scaling experiments."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    3 convolutional layers with configurable widths.
    
    Architecture:
    - Conv2d(3, w1, 3, padding=1) → BatchNorm → ReLU → MaxPool
    - Conv2d(w1, w2, 3, padding=1) → BatchNorm → ReLU → MaxPool
    - Conv2d(w2, w3, 3, padding=1) → BatchNorm → ReLU → MaxPool
    - Flatten → Linear(w3 * 4 * 4, num_classes)
    
    Default widths:
    - Standard: [32, 64, 128]
    - Lucas:    [29, 47, 76]
    """
    
    def __init__(self, widths: list = None, num_classes: int = 10):
        super().__init__()
        
        if widths is None:
            widths = [32, 64, 128]
        
        w1, w2, w3 = widths
        
        self.features = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, padding=1),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(w1, w2, kernel_size=3, padding=1),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(w2, w3, kernel_size=3, padding=1),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Linear(w3 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
