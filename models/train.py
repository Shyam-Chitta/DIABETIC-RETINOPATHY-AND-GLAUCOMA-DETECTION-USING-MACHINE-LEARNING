import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from collections import Counter

# ----------------------------
# Setup and Configuration
# ----------------------------

# Initialize directories
LOG_DIR = Path("models/logs")
CHECKPOINT_DIR = Path("models/checkpoints")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR/'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    IMAGE_DIR = "data/retina"  # Directory containing .png files
    CSV_PATH = "data/final_image1.csv"
    BATCH_SIZE = 32
    NUM_CLASSES = 5  # 0-4 severity levels
    EPOCHS = 30
    LR = 1e-4
    IMG_SIZE = 224
    
    # Class information
    CLASS_NAMES = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }
    
    # Augmentation parameters
    ROTATION = 15
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    HUE = 0.1

    @classmethod
    def verify_paths(cls):
        """Verify that all required paths exist"""
        logger.info("\nPath Verification:")
        logger.info(f"Image directory: {Path(cls.IMAGE_DIR).absolute()}")
        logger.info(f"Directory exists: {Path(cls.IMAGE_DIR).exists()}")
        
        if Path(cls.IMAGE_DIR).exists():
            sample_files = list(Path(cls.IMAGE_DIR).glob('*.*'))
            logger.info(f"Found {len(sample_files)} files in image directory")
            if len(sample_files) > 0:
                logger.info(f"First 5 files: {[f.name for f in sample_files[:5]]}")

        logger.info(f"\nCSV path: {Path(cls.CSV_PATH).absolute()}")
        logger.info(f"CSV exists: {Path(cls.CSV_PATH).exists()}")

# ----------------------------
# Dataset and Data Loading
# ----------------------------

class RetinaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.valid_indices = self._validate_files()
        
        if len(self.valid_indices) == 0:
            available_files = list(Path(img_dir).glob('*.*'))
            raise ValueError(
                f"No valid images found in dataset.\n"
                f"Image directory contains: {[f.name for f in available_files[:5]] + ['...'] if len(available_files) > 5 else []}"
            )
            
        self.class_weights = self._calculate_weights()
        logger.info(f"Class weights: {self.class_weights}")

    def _validate_files(self):
        """Check which files actually exist and return valid indices"""
        valid_indices = []
        missing_files = []
        
        for idx in range(len(self.df)):
            img_path = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing_files.append(self.df.iloc[idx]['filename'])
                
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} image files (keeping {len(valid_indices)} valid)")
            logger.info(f"First 5 missing files: {missing_files[:5]}")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_path = os.path.join(self.img_dir, self.df.iloc[actual_idx]['filename'])
        label = int(self.df.iloc[actual_idx]['label_idx'])  # Ensure Python int
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'label_tensor': torch.tensor(label, dtype=torch.long),
                'label_value': label  # Regular Python value for weight calculation
            }
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
            raise RuntimeError(f"Failed to load {img_path}")

    def _calculate_weights(self):
        valid_labels = [int(self.df.iloc[idx]['label_idx']) for idx in self.valid_indices]
        class_counts = Counter(valid_labels)
        total = sum(class_counts.values())
        return {cls: total/count for cls, count in class_counts.items()}

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(Config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(Config.ROTATION),
            transforms.ColorJitter(
                brightness=Config.BRIGHTNESS,
                contrast=Config.CONTRAST,
                saturation=Config.SATURATION,
                hue=Config.HUE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# ----------------------------
# Model and Training
# ----------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Ensure targets are long type
        if targets.dtype != torch.long:
            targets = targets.long()
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

def initialize_model():
    # Using modern weights API
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, Config.NUM_CLASSES)
    )
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    best_acc = 0.0
    device = next(model.parameters()).device
    
    for epoch in range(Config.EPOCHS):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label_tensor'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label_tensor'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)
        
        # Log results
        logger.info(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = CHECKPOINT_DIR / 'best_model.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved new best model (Acc: {val_acc:.2f}%) to {model_path}")

# ----------------------------
# Main Execution
# ----------------------------

def main():
    try:
        # Verify paths before starting
        Config.verify_paths()
        
        logger.info("Starting training process")
        
        # 1. Load and prepare data
        df = pd.read_csv(Config.CSV_PATH)
        logger.info(f"Data loaded from {Config.CSV_PATH}")
        
        # Create filename and validate labels
        df['filename'] = df['id_code'] + '.png'  # Change to .jpg if needed
        df['label_idx'] = df['diagnosis'].astype('int64')  # Ensure 64-bit integers
        
        # Verify first 5 files exist
        logger.info("\nVerifying first 5 files:")
        for i in range(min(5, len(df))):
            img_path = os.path.join(Config.IMAGE_DIR, df.iloc[i]['filename'])
            logger.info(f"{img_path} exists: {os.path.exists(img_path)}")

        # Filter valid classes
        valid_classes = [0, 1, 2, 3, 4]
        df = df[df['label_idx'].isin(valid_classes)]
        logger.info(f"Class distribution:\n{df['diagnosis'].value_counts()}")

        # Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, 
            stratify=df['label_idx'],
            random_state=42
        )

        # 2. Create datasets and dataloaders
        train_dataset = RetinaDataset(train_df, Config.IMAGE_DIR, get_transforms(True))
        val_dataset = RetinaDataset(val_df, Config.IMAGE_DIR, get_transforms(False))
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        # Create weighted sampler
        try:
            sample_weights = [train_dataset.class_weights[label] 
                            for label in [item['label_value'] for item in train_dataset]]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        except KeyError as e:
            logger.error(f"Invalid label value encountered: {e}")
            logger.info(f"Valid class weights: {train_dataset.class_weights}")
            raise

        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE,
            sampler=sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE,
            shuffle=False, num_workers=4
        )

        # 3. Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = initialize_model().to(device)
        
        # 4. Loss and optimizer
        class_weights = torch.tensor(list(train_dataset.class_weights.values()), 
                                   dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

        # 5. Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error in training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()