import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ── 1. DEVICE ───────────────────────────────────────────────
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 2. DATA TRANSFORMS ──────────────────────────────────────
# Neural networks need all images the same size.
# We also "augment" training images (flip, rotate) so the model
# doesn't just memorize — it learns to generalize.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),         # ResNet expects 224x224
    transforms.RandomHorizontalFlip(),     # randomly mirror the image
    transforms.RandomRotation(10),         # rotate up to 10 degrees
    transforms.ToTensor(),                 # convert image to numbers (0-1)
    transforms.Normalize([0.485, 0.456, 0.406],   # standard ImageNet mean
                         [0.229, 0.224, 0.225])    # standard ImageNet std
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 3. LOAD DATA ─────────────────────────────────────────────
# ImageFolder automatically reads subfolders as class labels
# chest_xray/train/NORMAL/ → class 0
# chest_xray/train/PNEUMONIA/ → class 1
train_dataset = datasets.ImageFolder("chest_xray/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("chest_xray/val",   transform=val_transform)
test_dataset  = datasets.ImageFolder("chest_xray/test",  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"Classes: {train_dataset.classes}")  # ['NORMAL', 'PNEUMONIA']
print(f"Training samples: {len(train_dataset)}")

# ── 4. MODEL ─────────────────────────────────────────────────
# ResNet18 is a CNN pretrained on 1.2 million images (ImageNet).
# "Transfer learning" = we borrow its learned features and just
# retrain the last layer for OUR task (Normal vs Pneumonia).
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all layers — we don't want to change the pretrained weights
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer: ResNet outputs 1000 classes, we need 2
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# ── 5. LOSS + OPTIMIZER ──────────────────────────────────────
# CrossEntropyLoss = standard loss for classification
# Adam = smart gradient descent optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# ── 6. TRAINING LOOP ─────────────────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()           # clear old gradients
        outputs = model(images)         # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # backpropagation
        optimizer.step()                # update weights
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():               # no gradient needed for eval
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

# ── 7. RUN TRAINING ──────────────────────────────────────────
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

# ── 8. TEST + SAVE ───────────────────────────────────────────
test_acc = evaluate(model, test_loader)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

torch.save(model.state_dict(), "pneumonia_model.pth")
print("Model saved as pneumonia_model.pth")
