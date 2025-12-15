"""
NutriVision AI - Model Evaluation Script
Computes offline metrics for academic presentation (not displayed in UI)
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import json
from tqdm import tqdm
from src.predict import load_predictor


def evaluate_model(validation_dir='data/validation', output_file='evaluation_results.json'):
    """
    Evaluate model on validation set and compute metrics
    
    Metrics computed:
    1. Top-1 Accuracy
    2. Top-3 Accuracy
    3. Confusion Matrix
    4. Precision, Recall, F1-score (macro averaged)
    5. Average Confidence
    
    Args:
        validation_dir: Directory containing validation images organized by class
        output_file: JSON file to save results
    """
    print("="*80)
    print("NUTRIVISION AI - MODEL EVALUATION")
    print("="*80)
    
    # Load predictor
    print("\n[1/5] Loading model...")
    predictor = load_predictor()
    class_names = predictor.class_names
    print(f"✓ Model loaded with {len(class_names)} classes")
    
    # Collect validation images
    print("\n[2/5] Collecting validation images...")
    validation_path = Path(validation_dir)
    
    if not validation_path.exists():
        print(f"✗ Validation directory not found: {validation_dir}")
        print("Please organize validation images in: data/validation/<class_name>/<image>.jpg")
        return
    
    image_paths = []
    true_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = validation_path / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            image_paths.extend(images)
            true_labels.extend([class_idx] * len(images))
            print(f"  {class_name}: {len(images)} images")
    
    if len(image_paths) == 0:
        print("✗ No validation images found")
        return
    
    print(f"\n✓ Total validation images: {len(image_paths)}")
    
    # Run predictions
    print("\n[3/5] Running predictions...")
    predicted_labels = []
    top3_predictions = []
    confidences = []
    
    for img_path in tqdm(image_paths, desc="Predicting"):
        result = predictor.predict(str(img_path), return_top_k=3, use_tta=True)
        
        # Top-1 prediction
        pred_class = result['predicted_class']
        pred_idx = class_names.index(pred_class)
        predicted_labels.append(pred_idx)
        
        # Top-3 predictions
        top3_classes = [class_names.index(cls) for cls, _ in result['all_predictions'][:3]]
        top3_predictions.append(top3_classes)
        
        # Confidence
        confidences.append(result['confidence'])
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    confidences = np.array(confidences)
    
    # Compute metrics
    print("\n[4/5] Computing metrics...")
    
    # 1. Top-1 Accuracy
    top1_accuracy = accuracy_score(true_labels, predicted_labels)
    
    # 2. Top-3 Accuracy
    top3_correct = sum([true_labels[i] in top3_predictions[i] for i in range(len(true_labels))])
    top3_accuracy = top3_correct / len(true_labels)
    
    # 3. Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # 4. Precision, Recall, F1-score (macro averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    
    # 5. Average Confidence
    avg_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nTop-1 Accuracy:     {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"Top-3 Accuracy:     {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    print(f"\nPrecision (macro):  {precision:.4f}")
    print(f"Recall (macro):     {recall:.4f}")
    print(f"F1-Score (macro):   {f1:.4f}")
    print(f"\nAvg Confidence:     {avg_confidence:.4f} ± {std_confidence:.4f}")
    print(f"Min Confidence:     {np.min(confidences):.4f}")
    print(f"Max Confidence:     {np.max(confidences):.4f}")
    
    # Save results
    print(f"\n[5/5] Saving results to {output_file}...")
    results = {
        'top1_accuracy': float(top1_accuracy),
        'top3_accuracy': float(top3_accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        'avg_confidence': float(avg_confidence),
        'std_confidence': float(std_confidence),
        'min_confidence': float(np.min(confidences)),
        'max_confidence': float(np.max(confidences)),
        'num_samples': len(image_paths),
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': class_names
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    evaluate_model()

