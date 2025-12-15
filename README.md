# NutriVision AI

**AI-Powered Nutritional Analysis for Indian Food**

A university final project that uses deep learning (EfficientNet-B1) to classify Indian food images and provide comprehensive nutritional insights, burn-off calculations, sustainability scores, and AI-generated cultural context.

---

## 🎯 Features

- **AI Food Recognition**: EfficientNet-B1 model trained on 18 Indian food classes
- **Nutrition Analysis**: Complete breakdown of calories, macros, fiber, and serving sizes
- **Burn-off Calculator**: Running minutes and Surya Namaskar rounds to burn calories
- **Sustainability Score**: Environmental impact rating (1-10 scale)
- **Cultural Context**: AI-generated explanations of food origins and significance
- **Leftover Ideas**: Creative suggestions for repurposing leftovers
- **Portion Control**: Dynamic portion size adjustment (0.5x to 3.0x)
- **Human-in-the-Loop**: Manual correction of predictions
- **Demo Mode**: Test without trained model using pre-loaded examples

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- Flask 2.3+
- 18 Indian food classes dataset
- (Optional) Groq or OpenAI API key for GenAI features

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd "NutriVision AI"

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Prepare Dataset

**⚠️ IMPORTANT: The raw dataset is NOT included in this repository (1.3 GB).**

You need to prepare your dataset before training:

```bash
# Create the data/raw directory structure (already exists with .gitkeep)
# Place your food images in the following structure:

data/raw/
├── burger/           # Place burger images here
├── butter_naan/      # Place butter naan images here
├── chai/             # Place chai images here
├── chole_bhature/    # Place chole bhature images here
├── dhokla/           # Place dhokla images here
├── fried_rice/       # Place fried rice images here
├── idli/             # Place idli images here
├── jalebi/           # Place jalebi images here
├── kaathi_rolls/     # Place kaathi rolls images here
├── kadai_paneer/     # Place kadai paneer images here
├── kulfi/            # Place kulfi images here
├── masala_dosa/      # Place masala dosa images here
├── momos/            # Place momos images here
├── paani_puri/       # Place paani puri images here
├── pakode/           # Place pakode images here
├── pav_bhaji/        # Place pav bhaji images here
├── pizza/            # Place pizza images here
└── samosa/           # Place samosa images here
```

**Dataset Requirements:**
- **18 food classes** (listed above)
- **Recommended**: 200-500 images per class
- **Format**: JPG, JPEG, or PNG
- **Total size**: ~1.3 GB (varies based on image count)

**Where to get the dataset:**
- Use your own collected images
- Download from Kaggle (search for "Indian Food Dataset")
- Use Google Images with appropriate licensing
- Contact the project author for dataset recommendations

### 3. Configure Environment (Optional)

Create a `.env` file for GenAI features:

```bash
cp .env.example .env
# Edit .env and add your API keys:
# GROQ_API_KEY=your_groq_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Train Model (First Time Only)

**⚠️ IMPORTANT: The trained model is NOT included in this repository (26 MB).**

You must train the model before using the full application:

```bash
# Train the EfficientNet-B1 model
python src/train.py

# This will:
# - Split data into train/val (80/20)
# - Train for 20 epochs
# - Save model to models/efficientnet_b1_indian.pth
# - Save checkpoints to models/checkpoints/
# - Takes ~30-60 minutes on CPU (faster on GPU)
```

**Training Output:**
- **Model file**: `models/efficientnet_b1_indian.pth` (~26 MB)
- **Checkpoints**: `models/checkpoints/checkpoint_epoch_*.pth` (~26 MB each)
- **Processed data**: `data/processed/train/` and `data/processed/val/` (~1.3 GB total)

**Note**: All model files and processed data are excluded from Git (see `.gitignore`).

### 4. Run Application

```bash
# Start Flask server
python app.py

# Server will start on http://127.0.0.1:5001
```

### 5. Access Application

Open your browser and navigate to:
- **Home**: http://127.0.0.1:5001
- **Scan Food**: http://127.0.0.1:5001/scan
- **Debug**: http://127.0.0.1:5001/debug
- **Demo**: http://127.0.0.1:5001/demo

---

## 🎮 Demo Mode

If you haven't trained the model yet, you can still explore the application:

1. Visit http://127.0.0.1:5001/demo
2. Select from 5 pre-loaded demo foods:
   - Samosa
   - Idli
   - Masala Dosa
   - Chole Bhature
   - Pav Bhaji
3. See complete analysis without model inference

---

## 📁 Project Structure

```
NutriVision AI/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
├── data/
│   ├── raw/                   # 18 food class directories
│   ├── processed/             # Train/val splits
│   └── food_data.json         # Nutrition database
├── models/
│   └── efficientnet_b1_indian.pth  # Trained model
├── src/
│   ├── train.py               # Training script
│   ├── predict.py             # Inference module
│   ├── utils.py               # Data utilities
│   ├── genai_helper.py        # GenAI integration
│   └── demo_config.py         # Demo configuration
├── static/
│   ├── uploads/               # User uploaded images
│   ├── demo_images/           # Demo images
│   ├── css/                   # Bootstrap styles
│   ├── js/                    # JavaScript
│   └── img/                   # Template images
└── templates/
    ├── base.html              # Base layout
    ├── index.html             # Home page
    ├── scan.html              # Upload page
    ├── results.html           # Results page
    └── debug.html             # Debug page
```

---

## 🍛 Supported Foods (18 Classes)

1. Samosa
2. Idli
3. Masala Dosa
4. Chole Bhature
5. Pav Bhaji
6. Paani Puri
7. Butter Naan
8. Kadai Paneer
9. Dhokla
10. Pakode
11. Momos
12. Kaathi Rolls
13. Fried Rice
14. Jalebi
15. Kulfi
16. Chai
17. Burger
18. Pizza

---

## 🔧 Troubleshooting

### Model Not Found
```bash
# Train the model first
python src/train.py
```

### GenAI Not Available
```bash
# Set API key in .env file
echo "GROQ_API_KEY=your_key_here" >> .env
```

### Upload Folder Missing
```bash
# Create upload folder
mkdir -p static/uploads
```

---

## 📦 Repository Structure & Excluded Files

### What's Included in This Repository

✅ **Source Code** (~500 KB)
- `src/` - All Python modules (train, predict, utils, genai_helper, demo_config)
- `app.py` - Flask application
- `templates/` - Jinja2 HTML templates
- `requirements.txt` - Python dependencies

✅ **Static Assets** (~5 MB)
- `static/css/` - Bootstrap 5 styles
- `static/js/` - JavaScript libraries
- `static/lib/` - Third-party libraries
- `static/img/` - Template images
- `static/demo_images/` - 5 demo food images (3.2 MB)

✅ **Data Files** (~2 MB)
- `data/food_data.json` - Nutrition database for 18 foods
- `.gitkeep` files - Preserve directory structure

✅ **Documentation** (~50 KB)
- `README.md` - This file
- `.env.example` - Environment variables template
- `run.sh` - Quick start script

✅ **Configuration** (~10 KB)
- `.gitignore` - Git exclusion rules

**Total Repository Size: ~15-20 MB** (GitHub-friendly!)

---

### What's Excluded from This Repository

The following files are excluded via `.gitignore` to keep the repository size manageable:

❌ **Raw Dataset** (~1.3 GB)
- `data/raw/*` - 18 food class directories with images
- **Reason**: Too large for GitHub (1.3 GB)
- **Action**: You must provide your own dataset (see "Prepare Dataset" section)

❌ **Processed Data** (~1.3 GB)
- `data/processed/train/*` - Training split (80%)
- `data/processed/val/*` - Validation split (20%)
- **Reason**: Generated during training, too large for GitHub
- **Action**: Automatically created when you run `python src/train.py`

❌ **Trained Models** (~26 MB + checkpoints)
- `models/efficientnet_b1_indian.pth` - Final trained model (26 MB)
- `models/checkpoints/*.pth` - Training checkpoints (~26 MB each)
- **Reason**: Large binary files, can be regenerated
- **Action**: Train the model using `python src/train.py`

❌ **Virtual Environment** (~500 MB)
- `venv/` - Python virtual environment
- **Reason**: Environment-specific, large size
- **Action**: Create using `python3 -m venv venv`

❌ **User Uploads** (varies)
- `static/uploads/*` - User-uploaded food images
- **Reason**: User-generated content, not part of source code
- **Action**: Automatically created when users upload images

❌ **Temporary Files**
- `__pycache__/` - Python bytecode cache
- `*.pyc`, `*.pyo` - Compiled Python files
- `.DS_Store` - macOS metadata
- `.env` - Environment variables (contains API keys)
- **Reason**: Temporary/sensitive files
- **Action**: Automatically generated or user-specific

---

### First-Time Setup Checklist

After cloning this repository, you need to:

1. ✅ **Create virtual environment**: `python3 -m venv venv`
2. ✅ **Install dependencies**: `pip install -r requirements.txt`
3. ✅ **Prepare dataset**: Place images in `data/raw/<food_class>/`
4. ✅ **Train model**: `python src/train.py` (30-60 minutes)
5. ✅ **(Optional) Configure GenAI**: Copy `.env.example` to `.env` and add API keys
6. ✅ **Run application**: `python app.py` or `./run.sh`

**OR** skip training and use **Demo Mode**: Visit `http://127.0.0.1:5001/demo`

---

## 📊 Technical Details

- **Model**: EfficientNet-B1 (pretrained on ImageNet)
- **Training**: Transfer learning with frozen backbone
- **Input Size**: 224×224 RGB
- **Batch Size**: 32
- **Epochs**: 20
- **Optimizer**: Adam (lr=0.001)
- **Augmentation**: Albumentations (rotation, flip, brightness)
- **Framework**: PyTorch + Flask
- **Frontend**: Bootstrap 5 + Jinja2
- **GenAI**: Groq (LLaMA-3.1-70b) or OpenAI (GPT-3.5)

---

## 👨‍💻 Author

University Final Project - MSc ADSA

---

## 📄 License

Educational use only.

