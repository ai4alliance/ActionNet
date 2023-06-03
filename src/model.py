from torch import nn
from torchvision import models


class ActionClassifier(nn.Module):
    def __init__(self, train_last_nlayer, hidden_size, dropout, ntargets):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=True)
        modules = list(resnet.children())[:-1] # delete last layer

        self.resnet = nn.Sequential(*modules)
        for param in self.resnet[:-train_last_nlayer].parameters():
            param.requires_grad = False
            
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.Dropout(dropout),
            nn.Linear(resnet.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, ntargets),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

