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

# Demo prediction results (for when model is not available)
DEMO_PREDICTIONS = {
    'samosa': {
        'predicted_class': 'samosa',
        'confidence': 0.92,
        'all_predictions': [
            ('samosa', 0.92),
            ('pakode', 0.05),
            ('kaathi_rolls', 0.02)
        ],
        'low_confidence_warning': False,
        'demo_mode': True
    },
    'idli': {
        'predicted_class': 'idli',
        'confidence': 0.95,
        'all_predictions': [
            ('idli', 0.95),
            ('dhokla', 0.03),
            ('masala_dosa', 0.01)
        ],
        'low_confidence_warning': False,
        'demo_mode': True
    },
    'masala_dosa': {
        'predicted_class': 'masala_dosa',
        'confidence': 0.88,
        'all_predictions': [
            ('masala_dosa', 0.88),
            ('idli', 0.07),
            ('fried_rice', 0.03)
        ],
        'low_confidence_warning': False,
        'demo_mode': True
    },
    'chole_bhature': {
        'predicted_class': 'chole_bhature',
        'confidence': 0.91,
        'all_predictions': [
            ('chole_bhature', 0.91),
            ('pav_bhaji', 0.05),
            ('butter_naan', 0.02)
        ],
        'low_confidence_warning': False,
        'demo_mode': True
    },
    'pav_bhaji': {
        'predicted_class': 'pav_bhaji',
        'confidence': 0.89,
        'all_predictions': [
            ('pav_bhaji', 0.89),
            ('chole_bhature', 0.06),
            ('kadai_paneer', 0.03)
        ],
        'low_confidence_warning': False,
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

