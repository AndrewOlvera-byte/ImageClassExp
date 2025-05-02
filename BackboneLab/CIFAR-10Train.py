import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from Archs.ResNetBackbone import ResNetBackbone
from Archs.VisionArch import VisionArch
import multiprocessing as mp
from multiprocessing import freeze_support
mp.set_start_method('spawn', force=True)

#Hyperparameters
batch_size = 128
num_classes = 10
use_mixup = False
use_cutmix = False
mixup_alpha = 0.2
use_amp = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data transformations: CIFAR-10 mean/std for normalization
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

#CIFAR-10 training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize,

])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

#CIFAR-10 datasets
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

#Data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

#Initialize model
backbone = ResNetBackbone()
model = VisionArch(backbone, num_classes=num_classes)
model.to(device)

#Loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#Optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
epochs = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

#EMA weight tracker
ema_decay = 0.999
ema_model_params = [p.clone().detach() for p in model.parameters()]

def rand_bbox(size, lam):
    """
    Generate a random rectangular bounding box for CutMix
    """
    #size: (batch_size, channels, H, W)
    B, C, H, W = size
    cut_ratio = np.sqrt(1.0-lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    #Random position
    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)

    #Compute box corners
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    """
    Training loop for one epoch
    """
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        #Apply MixUp or CutMix augmentation if enabled
        if use_mixup or use_cutmix:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(images.size(0)).to(device)
            if use_mixup:
                mixed_images = lam * images + (1 - lam) * images[perm]
                target_a, target_b = targets, targets[perm]
            elif use_cutmix:
                x1, y1, x2, y2 = rand_bbox(images.size(), lam)
                mixed_images = images.clone()
                mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
                lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
                target_a, target_b = targets, targets[perm]
            images = mixed_images
        else:
            lam = 1.0 # No augmentation

        # Forward pass
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            if use_mixup or use_cutmix:
                loss = criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
            else:
                loss = criterion(outputs, targets)
        running_loss += loss.item()

        #Backpropogation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        #EMA update
        with torch.no_grad():
            for p, ema_p in zip(model.parameters(), ema_model_params):
                ema_p.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        avg_loss = running_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

#Function to load EMA weights into the model for evaluation
def load_ema_weights(model, ema_params):
    for p, ema_p in zip(model.parameters(), ema_params):
        p.data.copy_(ema_p.data)


def main():
    #Training loop
    best_accuracy = 0.0

    print("Training...")

    #Training loop
    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch} of {epochs}")
        train_one_epoch(model, train_loader, optimizer, scaler, epoch)

        #Evaluate on test set using EMA weights for stability
        current_weights = [p.data.clone() for p in model.parameters()]
        load_ema_weights(model, ema_model_params)
        val_acc = evaluate(model, test_loader)
        
        #Restore original weights
        for p, cw in zip(model.parameters(), current_weights):
            p.data.copy_(cw)
        print(f"Epoch {epoch} completed. Validation Accuracy: {val_acc:.2f}%")

        #Save checkpoint if improved accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), f'./models/best_model_resnet_no_aug.pth')

    print(f"Training complete. Best Validation Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    freeze_support()
    main()
