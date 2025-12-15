"""
NutriVision AI - Flask Web Application
Main application file for the NutriVision AI food analysis system
"""

import os
import secrets
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session

# Import our modules
from src.predict import load_predictor
from src.utils import load_food_data_manager
from src.genai_helper import load_genai_helper
from src.demo_config import (
    get_demo_prediction,
    get_demo_image,
    is_demo_mode,
    get_available_demo_foods,
    validate_demo_image
)


# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Initialize AI components (lazy loading)
predictor = None
food_manager = None
genai_helper = None


def get_predictor():
    """Lazy load predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = load_predictor()
        except FileNotFoundError:
            print("⚠️  Model not found. Please train the model first.")
            predictor = None
    return predictor


def get_food_manager():
    """Lazy load food data manager"""
    global food_manager
    if food_manager is None:
        food_manager = load_food_data_manager()
    return food_manager


def get_genai_helper():
    """Lazy load GenAI helper"""
    global genai_helper
    if genai_helper is None:
        genai_helper = load_genai_helper(api_provider='groq')
    return genai_helper


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/scan')
def scan():
    """Scan page - image upload interface"""
    # Check if model is available
    pred = get_predictor()
    model_available = pred is not None
    demo_foods = get_available_demo_foods() if not model_available else []

    return render_template('scan.html',
                         model_available=model_available,
                         demo_foods=demo_foods)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint
    Handles image upload, prediction, and nutrition analysis
    """
    try:
        # Check if file was uploaded
        if 'food_image' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('scan'))
        
        file = request.files['food_image']
        
        # Check if file is empty
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('scan'))
        
        # Validate file
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
            return redirect(url_for('scan'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            flash('Model not available. Please train the model first.', 'error')
            return redirect(url_for('scan'))
        
        # Predict food class
        prediction_result = pred.predict(filepath, return_top_k=3)
        
        # Get nutrition data
        manager = get_food_manager()
        portion_multiplier = float(request.form.get('portion_multiplier', 1.0))
        nutrition_data = manager.get_nutrition_data(
            prediction_result['predicted_class'],
            portion_multiplier=portion_multiplier
        )
        
        # Get GenAI analysis
        genai = get_genai_helper()
        ai_analysis = genai.generate_complete_analysis(
            prediction_result['predicted_class'],
            nutrition_data
        )
        
        # Store results in session
        session['analysis_result'] = {
            'image_path': unique_filename,
            'prediction': prediction_result,
            'nutrition': nutrition_data,
            'ai_analysis': ai_analysis,
            'portion_multiplier': portion_multiplier,
            'timestamp': timestamp
        }
        
        return redirect(url_for('results'))
    
    except Exception as e:
        print(f"Error in analyze: {e}")
        flash(f'An error occurred during analysis: {str(e)}', 'error')
        return redirect(url_for('scan'))


@app.route('/results')
def results():
    """Display analysis results"""
    analysis_result = session.get('analysis_result')

    if not analysis_result:
        flash('No analysis results found. Please upload an image first.', 'warning')
        return redirect(url_for('scan'))

    return render_template('results.html', result=analysis_result)


@app.route('/api/update_portion', methods=['POST'])
def update_portion():
    """
    API endpoint to update portion size
    Returns updated nutrition data
    """
    try:
        data = request.get_json()
        food_name = data.get('food_name')
        portion_multiplier = float(data.get('portion_multiplier', 1.0))

        # Get updated nutrition data
        manager = get_food_manager()
        nutrition_data = manager.get_nutrition_data(food_name, portion_multiplier=portion_multiplier)

        return jsonify({
            'status': 'success',
            'nutrition': nutrition_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/correct_prediction', methods=['POST'])
def correct_prediction():
    """
    API endpoint for human-in-the-loop correction
    Allows user to manually correct the prediction
    """
    try:
        data = request.get_json()
        corrected_food = data.get('corrected_food')
        portion_multiplier = float(data.get('portion_multiplier', 1.0))

        # Get nutrition data for corrected food
        manager = get_food_manager()
        nutrition_data = manager.get_nutrition_data(corrected_food, portion_multiplier=portion_multiplier)

        if nutrition_data.get('status') == 'not_found':
            return jsonify({
                'status': 'error',
                'message': f'Food "{corrected_food}" not found in database'
            }), 404

        # Get GenAI analysis for corrected food
        genai = get_genai_helper()
        ai_analysis = genai.generate_complete_analysis(corrected_food, nutrition_data)

        # Update session
        if 'analysis_result' in session:
            session['analysis_result']['prediction']['predicted_class'] = corrected_food
            session['analysis_result']['prediction']['manually_corrected'] = True
            session['analysis_result']['nutrition'] = nutrition_data
            session['analysis_result']['ai_analysis'] = ai_analysis
            session.modified = True

        return jsonify({
            'status': 'success',
            'nutrition': nutrition_data,
            'ai_analysis': ai_analysis
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/food_list')
def food_list():
    """
    API endpoint to get list of all available foods
    Used for autocomplete/dropdown
    """
    try:
        manager = get_food_manager()
        foods = manager.get_all_food_names()

        # Format for display (replace underscores with spaces, capitalize)
        formatted_foods = [
            {
                'value': food,
                'label': food.replace('_', ' ').title()
            }
            for food in sorted(foods)
        ]

        return jsonify({
            'status': 'success',
            'foods': formatted_foods
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/debug')
def debug():
    """
    Debug page - shows system status and allows forced predictions
    """
    pred = get_predictor()
    manager = get_food_manager()
    genai = get_genai_helper()

    status = {
        'model_loaded': pred is not None,
        'food_data_loaded': manager is not None,
        'genai_available': genai.is_available() if genai else False,
        'food_count': len(manager.get_all_food_names()) if manager else 0,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': Path(app.config['UPLOAD_FOLDER']).exists()
    }

    return render_template('debug.html', status=status) if Path('templates/debug.html').exists() \
        else jsonify(status)


@app.route('/api/demo_predict', methods=['POST'])
def demo_predict():
    """
    Demo endpoint for forced prediction (for testing without model)
    """
    try:
        data = request.get_json()
        demo_food = data.get('food_name', 'samosa')
        portion_multiplier = float(data.get('portion_multiplier', 1.0))

        # Get nutrition data
        manager = get_food_manager()
        nutrition_data = manager.get_nutrition_data(demo_food, portion_multiplier=portion_multiplier)

        if nutrition_data.get('status') == 'not_found':
            return jsonify({
                'status': 'error',
                'message': f'Food "{demo_food}" not found'
            }), 404

        # Get GenAI analysis
        genai = get_genai_helper()
        ai_analysis = genai.generate_complete_analysis(demo_food, nutrition_data)

        # Get demo prediction from config
        demo_result = get_demo_prediction(demo_food)
        if not demo_result:
            # Fallback demo result
            demo_result = {
                'predicted_class': demo_food,
                'confidence': 0.95,
                'all_predictions': [
                    (demo_food, 0.95),
                    ('idli', 0.03),
                    ('dosa', 0.02)
                ],
                'low_confidence_warning': False,
                'demo_mode': True
            }

        return jsonify({
            'status': 'success',
            'prediction': demo_result,
            'nutrition': nutrition_data,
            'ai_analysis': ai_analysis
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/demo')
def demo():
    """
    Demo page - pre-loaded analysis without model
    Shows example results for demonstration purposes
    """
    # Get available demo foods
    demo_foods = get_available_demo_foods()

    # Default to samosa
    selected_food = request.args.get('food', 'samosa')
    if selected_food not in demo_foods:
        selected_food = 'samosa'

    # Get demo image
    demo_image = get_demo_image(selected_food)
    if not demo_image or not validate_demo_image(selected_food):
        flash('Demo image not available', 'warning')
        return redirect(url_for('index'))

    # Get demo prediction
    prediction_result = get_demo_prediction(selected_food)

    # Get nutrition data
    manager = get_food_manager()
    nutrition_data = manager.get_nutrition_data(selected_food, portion_multiplier=1.0)

    # Get GenAI analysis
    genai = get_genai_helper()
    ai_analysis = genai.generate_complete_analysis(selected_food, nutrition_data)

    # Create demo result
    demo_result = {
        'image_path': demo_image.replace('static/', ''),  # Remove 'static/' prefix
        'prediction': prediction_result,
        'nutrition': nutrition_data,
        'ai_analysis': ai_analysis,
        'portion_multiplier': 1.0,
        'timestamp': 'demo',
        'is_demo': True
    }

    return render_template('results.html', result=demo_result, demo_foods=demo_foods)


@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('404.html') if Path('templates/404.html').exists() \
        else jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return render_template('500.html') if Path('templates/500.html').exists() \
        else jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def file_too_large(error):
    """413 error handler - file too large"""
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('scan'))


if __name__ == '__main__':
    # Development server
    print("="*60)
    print("NutriVision AI - Starting Flask Application")
    print("="*60)

    # Check system status
    pred = get_predictor()
    manager = get_food_manager()
    genai = get_genai_helper()

    print(f"✓ Food Data Manager: {'Loaded' if manager else 'Failed'}")
    print(f"✓ Model: {'Loaded' if pred else 'Not Found (train first)'}")
    print(f"✓ GenAI: {'Available' if genai and genai.is_available() else 'Not Available (set API key)'}")
    print(f"✓ Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print("="*60)
    print("Starting server on http://127.0.0.1:5001")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5001)

