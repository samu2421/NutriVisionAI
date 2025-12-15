"""
NutriVision AI - Inference/Prediction Module
Handles model loading and food image prediction
"""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path


class FoodPredictor:
    """Food image classifier using trained EfficientNet-B1"""
    
    def __init__(self, model_path='models/efficientnet_b1_indian.pth', device=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: torch device (defaults to CPU)
        """
        self.model_path = model_path
        self.device = device if device else torch.device('cpu')
        self.model = None
        self.class_names = None
        self.transform = self._get_transform()
        
        # Load model
        self.load_model()
    
    def _get_transform(self):
        """Get inference transforms (no augmentation)"""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def load_model(self):
        """Load trained model from checkpoint"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get class names
        self.class_names = checkpoint.get('class_names', [])
        num_classes = len(self.class_names)
        
        # Create model architecture
        model = models.efficientnet_b1(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        self.model = model
        
        print(f"✓ Model loaded from {self.model_path}")
        print(f"✓ Classes: {num_classes}")
        print(f"✓ Device: {self.device}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Load image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        else:
            raise ValueError("image_path must be a file path or PIL Image")
        
        # Convert to numpy
        image = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image_path, return_top_k=3):
        """
        Predict food class from image
        
        Args:
            image_path: Path to image file or PIL Image
            return_top_k: Number of top predictions to return
            
        Returns:
            dict with prediction results:
                - predicted_class: Top predicted class name
                - confidence: Confidence score (0-1)
                - all_predictions: List of (class_name, confidence) tuples
                - low_confidence_warning: Boolean flag if confidence < 0.6
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(return_top_k, len(self.class_names)))
        
        # Convert to CPU and numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Build results
        all_predictions = [
            (self.class_names[idx], float(prob))
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        predicted_class = all_predictions[0][0]
        confidence = all_predictions[0][1]
        
        # Low confidence warning
        low_confidence_warning = confidence < 0.6
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'low_confidence_warning': low_confidence_warning,
            'confidence_threshold': 0.6
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        return results


def load_predictor(model_path='models/efficientnet_b1_indian.pth'):
    """
    Convenience function to load predictor
    
    Args:
        model_path: Path to trained model
        
    Returns:
        FoodPredictor instance
    """
    return FoodPredictor(model_path=model_path)


if __name__ == "__main__":
    # Test the predictor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load predictor
    predictor = load_predictor()
    
    # Predict
    result = predictor.predict(image_path)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if result['low_confidence_warning']:
        print("⚠️  WARNING: Low confidence prediction!")
    
    print(f"\nTop {len(result['all_predictions'])} Predictions:")
    for i, (class_name, conf) in enumerate(result['all_predictions'], 1):
        print(f"  {i}. {class_name}: {conf:.2%}")
    print("="*50)

