"""
Demo Configuration and Safety Features
Provides fallback mechanisms for demo mode
"""

import os
from pathlib import Path

# Demo mode configuration
DEMO_MODE = os.getenv('DEMO_MODE', 'False').lower() == 'true'

# Demo images mapping
DEMO_IMAGES = {
    'samosa': 'static/demo_images/demo_samosa.jpg',
    'idli': 'static/demo_images/demo_idli.jpg',
    'masala_dosa': 'static/demo_images/demo_masala_dosa.jpg',
    'chole_bhature': 'static/demo_images/demo_chole_bhature.jpg',
    'pav_bhaji': 'static/demo_images/demo_pav_bhaji.jpg',
}

# Default demo food if model not available
DEFAULT_DEMO_FOOD = 'samosa'

# Precomputed predictions shown on /demo.
# Real model output: efficientnet_b1_indian.pth with TTA, run on the
# static/demo_images/demo_<class>.jpg files themselves (validation-set images).
DEMO_PREDICTIONS = {
    'samosa': {
        'predicted_class': 'samosa',
        'confidence': 0.9933,
        'all_predictions': [
            ('samosa', 0.9933),
            ('kulfi', 0.0035),
            ('momos', 0.0009)
        ],
        'low_confidence_warning': False,
        'tta_used': True,
        'demo_mode': True
    },
    'idli': {
        'predicted_class': 'idli',
        'confidence': 0.9963,
        'all_predictions': [
            ('idli', 0.9963),
            ('chole_bhature', 0.0029),
            ('masala_dosa', 0.0006)
        ],
        'low_confidence_warning': False,
        'tta_used': True,
        'demo_mode': True
    },
    'masala_dosa': {
        'predicted_class': 'masala_dosa',
        'confidence': 0.9982,
        'all_predictions': [
            ('masala_dosa', 0.9982),
            ('idli', 0.0008),
            ('kadai_paneer', 0.0004)
        ],
        'low_confidence_warning': False,
        'tta_used': True,
        'demo_mode': True
    },
    'chole_bhature': {
        'predicted_class': 'chole_bhature',
        'confidence': 0.9984,
        'all_predictions': [
            ('chole_bhature', 0.9984),
            ('butter_naan', 0.0011),
            ('pakode', 0.0001)
        ],
        'low_confidence_warning': False,
        'tta_used': True,
        'demo_mode': True
    },
    'pav_bhaji': {
        'predicted_class': 'pav_bhaji',
        'confidence': 0.9768,
        'all_predictions': [
            ('pav_bhaji', 0.9768),
            ('chole_bhature', 0.0161),
            ('samosa', 0.0019)
        ],
        'low_confidence_warning': False,
        'tta_used': True,
        'demo_mode': True
    }
}


def get_demo_image(food_name):
    """
    Get demo image path for a food item
    
    Args:
        food_name: Name of the food
        
    Returns:
        Path to demo image or None
    """
    return DEMO_IMAGES.get(food_name)


def get_demo_prediction(food_name):
    """
    Get demo prediction for a food item
    
    Args:
        food_name: Name of the food
        
    Returns:
        Demo prediction dict or None
    """
    return DEMO_PREDICTIONS.get(food_name)


def is_demo_mode():
    """Check if demo mode is enabled"""
    return DEMO_MODE


def get_available_demo_foods():
    """Get list of foods available in demo mode"""
    return list(DEMO_IMAGES.keys())


def validate_demo_image(food_name):
    """
    Validate that demo image exists
    
    Args:
        food_name: Name of the food
        
    Returns:
        True if image exists, False otherwise
    """
    image_path = get_demo_image(food_name)
    if image_path:
        return Path(image_path).exists()
    return False


def get_fallback_food():
    """Get fallback food when requested food not available"""
    return DEFAULT_DEMO_FOOD


def create_demo_prediction_result(food_name, use_fallback=False):
    """
    Create a complete demo prediction result
    
    Args:
        food_name: Name of the food
        use_fallback: If True, use fallback food if requested food not available
        
    Returns:
        Demo prediction dict
    """
    if food_name not in DEMO_PREDICTIONS:
        if use_fallback:
            food_name = get_fallback_food()
        else:
            return None
    
    return DEMO_PREDICTIONS.get(food_name)

