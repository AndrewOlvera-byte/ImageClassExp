import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=None):
        super(ClassificationHead, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_features  # by default, one linear layer
        # If a hidden_dim is provided, use two Linear layers (MLP head)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VisionArch(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super(VisionArch, self).__init__()
        self.backbone = backbone
        self.head = ClassificationHead(backbone.num_features, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits
