import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from tqdm import tqdm

# Step 1: Config
dataset_dir = r"C:\Users\heman\OneDrive\Documents\Python projects\Mini project-5 sem\Jarvis\project trail 1\skinnnn\Skin-Disease-Detection-master\sd-198\images"
img_size = 224
batch_size = 32
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)

# Step 2: Transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Step 3: Dataset and Dataloaders
full_dataset = ImageFolder(root=dataset_dir, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

# Compute class weights
targets = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Sampler to balance classes
sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Focal Loss (fixed __init__)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Step 5: Model
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# Step 6: Training Setup
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    top5_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Top-5 accuracy
            _, top5_preds = outputs.topk(5, dim=1)
            top5_correct += sum([labels[i] in top5_preds[i] for i in range(labels.size(0))])

    acc = correct / total
    top5_acc = top5_correct / total
    return acc, top5_acc

# Step 7: Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_acc, val_top5 = evaluate(model, val_loader)
    print(f"\nValidation Accuracy: {val_acc:.4f}, Top-5 Accuracy: {val_top5:.4f}\n")

# Step 8: Save model
model_path = r"C:\Users\heman\OneDrive\Documents\Python projects\Mini project-5 sem\Jarvis\project trail 1\skinnnn\Skin-Disease-Detection-master\efficientnet_sd198_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")
