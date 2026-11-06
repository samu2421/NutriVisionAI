"""
NutriVision AI - Model Training Script
Trains EfficientNet-B1 on 18 Indian food classes
"""

import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class FoodDataset(Dataset):
    """Custom Dataset for Indian Food Images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        return image, label


def get_transforms(is_train=True):
    """Get Albumentations transforms"""
    if is_train:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def prepare_data_splits(raw_data_dir, processed_dir, val_split=0.2, random_state=42):
    """Split raw data into train/val if processed directory is empty"""
    train_dir = Path(processed_dir) / 'train'
    val_dir = Path(processed_dir) / 'val'
    
    # Check if already processed (ignore dotfiles such as .gitkeep and .DS_Store)
    def has_entries(directory):
        return directory.exists() and any(not p.name.startswith('.') for p in directory.iterdir())

    if has_entries(train_dir) and has_entries(val_dir):
        print("Data already split. Using existing train/val directories.")
        return
    
    print("Splitting data into train/val...")
    raw_path = Path(raw_data_dir)
    class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_name}")
            continue
        
        # Split
        train_imgs, val_imgs = train_test_split(images, test_size=val_split, random_state=random_state)
        
        # Create class directories
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img in train_imgs:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir / class_name / img.name)
        
        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val")


def load_data_from_directory(data_dir):
    """Load image paths and labels from directory structure"""
    data_path = Path(data_dir)
    image_paths = []
    labels = []
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = data_path / class_name
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])
    
    return image_paths, labels, class_names


def create_model(num_classes=18):
    """Create EfficientNet-B1 with frozen backbone"""
    # Load pretrained EfficientNet-B1
    model = models.efficientnet_b1(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, num_classes)
    )
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    """Main training function"""
    # Configuration
    RAW_DATA_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    MODEL_SAVE_PATH = 'models/efficientnet_b1_indian.pth'
    CHECKPOINT_DIR = 'models/checkpoints'

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 18

    # Device
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Prepare data splits
    prepare_data_splits(RAW_DATA_DIR, PROCESSED_DIR)

    # Load data
    print("\nLoading training data...")
    train_paths, train_labels, class_names = load_data_from_directory(f'{PROCESSED_DIR}/train')
    print(f"Training samples: {len(train_paths)}")

    print("\nLoading validation data...")
    val_paths, val_labels, _ = load_data_from_directory(f'{PROCESSED_DIR}/val')
    print(f"Validation samples: {len(val_paths)}")

    print(f"\nClasses ({len(class_names)}): {class_names}")

    # Create datasets
    train_dataset = FoodDataset(train_paths, train_labels, transform=get_transforms(is_train=True))
    val_dataset = FoodDataset(val_paths, val_labels, transform=get_transforms(is_train=False))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("\nCreating EfficientNet-B1 model...")
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, MODEL_SAVE_PATH)
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()

