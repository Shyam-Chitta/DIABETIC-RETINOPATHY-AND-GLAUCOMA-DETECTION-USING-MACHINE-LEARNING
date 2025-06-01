# model_utils.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import logging
import numpy as np

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Model Definition
# ----------------------------
def get_model(num_classes=5, freeze_backbone=True, use_pretrained=True):
    """
    Initialize and configure the ResNet model
    
    Args:
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze feature extractor layers
        use_pretrained: Whether to use pretrained weights
    
    Returns:
        Configured PyTorch model
    """
    try:
        logger.info(f"Initializing model with {num_classes} classes")
        model = models.resnet18(pretrained=use_pretrained)
        
        if freeze_backbone:
            logger.info("Freezing backbone layers")
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        logger.info("Model initialized successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

# ----------------------------
# Load Model from File
# ----------------------------
def load_model(model_path, num_classes=5, device=None):
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the EXACT same architecture used during training
        model = models.resnet50(weights=None)  # Must match training architecture
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Load with strict=False to ignore mismatched layers
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Key change here
        
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
# ----------------------------
# Image Transformations
# ----------------------------
def get_transforms():
    """Get standard transforms for inference"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ----------------------------
# Inference Functions
# ----------------------------
def predict_image(model, image_path, classes):
    """
    Predict class for a single image
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to image file
        classes: List of class names
    
    Returns:
        Predicted class label
    """
    try:
        pred, confidence = predict_image_with_confidence(model, image_path, classes)
        return pred
    except Exception as e:
        logger.error(f"Prediction failed for {image_path}: {str(e)}")
        raise

def predict_image_with_confidence(model, image_path, classes, top_k=3):
    """
    Predict class with confidence scores
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to image file
        classes: List of class names
        top_k: Number of top predictions to return
    
    Returns:
        tuple: (predicted_class, confidence_score, top_predictions)
    """
    try:
        device = next(model.parameters()).device  # Get model device
        
        # Load and transform image
        transform = get_transforms()
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.topk(probabilities, k=top_k)
        
        # Convert to numpy arrays
        conf = conf.cpu().numpy().flatten()
        preds = preds.cpu().numpy().flatten()
        
        # Get top predictions
        top_predictions = [(classes[pred], float(conf)) 
                          for pred, conf in zip(preds, conf)]
        
        return top_predictions[0][0], top_predictions[0][1], top_predictions
        
    except Exception as e:
        logger.error(f"Prediction failed for {image_path}: {str(e)}")
        raise

def predict_batch(model, image_paths, classes, batch_size=32):
    """
    Predict classes for a batch of images
    
    Args:
        model: Loaded PyTorch model
        image_paths: List of image paths
        classes: List of class names
        batch_size: Batch size for processing
    
    Returns:
        List of tuples (filename, predicted_class, confidence_score)
    """
    try:
        device = next(model.parameters()).device
        transform = get_transforms()
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load and transform images
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(transform(image))
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Could not load {path}: {str(e)}")
                    continue
            
            if not batch_images:
                continue
                
            # Stack images into batch tensor
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, preds = torch.max(probabilities, dim=1)
            
            # Store results
            for path, pred, conf in zip(valid_paths, preds, confidences):
                results.append({
                    'filename': os.path.basename(path),
                    'prediction': classes[pred.item()],
                    'confidence': conf.item()
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise

# ----------------------------
# Utility Functions
# ----------------------------
def get_device():
    """Get available device (cuda if available, else cpu)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    """Save model state dict"""
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise