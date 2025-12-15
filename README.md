# NutriVision AI

**Intelligent Food Recognition and Nutritional Analysis System for Indian Cuisine**

A production-ready web application that combines computer vision (EfficientNet-B1), structured nutrition databases, and large language models to provide comprehensive food analysis including calorie estimation, macro-nutrient breakdown, sustainability scoring, and culturally-aware recipe suggestions.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)
- Groq API key (optional, for GenAI features)

### Installation Steps

**1. Clone the repository**
```bash
git clone <repository-url>
cd "NutriVision AI"
```

**2. Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables** (optional)

Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

> **Note:** The application works without the API key, but GenAI features (cultural explanations, recipe ideas, AI insights) will use fallback static content.

**5. Run the Flask application**
```bash
python app.py
```

**6. Access the application**

Open your browser and navigate to:
```
http://127.0.0.1:5001
```

---

## 📂 Project Directory Tree

```
NutriVision AI/
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create manually)
├── README.md                      # Project documentation
│
├── src/                           # Source code modules
│   ├── __init__.py               # Package initializer
│   ├── train.py                  # Model training script
│   ├── predict.py                # Inference engine with TTA
│   ├── utils.py                  # Data utilities & FoodDataManager
│   ├── genai_helper.py           # Groq API integration
│   ├── demo_config.py            # Demo mode configuration
│   └── evaluate_model.py         # Offline evaluation metrics
│
├── models/                        # Model storage
│   ├── efficientnet_b1_indian.pth  # Trained model (26 MB)
│   └── checkpoints/              # Training checkpoints
│
├── data/                          # Data storage
│   ├── food_data.json            # Nutrition database (18 foods)
│   ├── raw/                      # Raw dataset (18 class folders)
│   └── processed/                # Train/val splits (auto-generated)
│
├── static/                        # Static assets
│   ├── uploads/                  # User-uploaded images
│   ├── demo_images/              # Demo food images
│   ├── css/                      # Bootstrap 5 styles
│   ├── js/                       # JavaScript libraries
│   ├── lib/                      # Third-party libraries
│   └── img/                      # Template images
│
└── templates/                     # Jinja2 HTML templates
    ├── base.html                 # Base layout
    ├── index.html                # Home page
    ├── scan.html                 # Upload interface
    ├── results.html              # Results page (4 tabs)
    ├── debug.html                # System debug page
    ├── 404.html                  # Error page
    └── 500.html                  # Server error page
```

---

## 🧩 Features

### Core Vision Intelligence
- **Deep Learning Classification**: EfficientNet-B1 model trained on 18 Indian food classes
- **Test Time Augmentation (TTA)**: Logit averaging across original and horizontally flipped images for improved confidence
- **ImageNet Preprocessing**: Standard SmallestMaxSize(256) + CenterCrop(224) pipeline to reduce aspect ratio distortion
- **Confidence Calibration**: Honest uncertainty communication with domain shift awareness

### Generative Intelligence
- **Cultural Explanations**: AI-generated context about food origins, Ayurvedic properties, and regional significance
- **Leftover Ideas**: Creative suggestions for repurposing leftovers
- **Flavor Pairing**: Recommended beverage and side dish pairings
- **AI Insights**: Structured 3-bullet summaries grounded in nutrition data

### Analytical Intelligence
- **Nutrition Breakdown**: Calories, protein, carbs, fats, fiber per serving
- **Portion Simulator**: Dynamic scaling (0.5× to 3.0×) with real-time updates
- **Macro Radar Chart**: Interactive Chart.js visualization comparing meal to ideal balanced ratios
- **Burn-Off Calculator**: Running minutes, cycling minutes, and Surya Namaskar rounds

### Sustainability Intelligence
- **Eco-Score**: 1-10 rating based on carbon footprint, water usage, and food miles
- **Eco-Badge**: Color-coded visual indicator (green/yellow/red)
- **Sustainability Explanations**: AI-generated insights on environmental impact

### User Experience
- **4-Tab Interface**: Overview, Nutrition Lab, Chef's Kitchen, Eco-Impact
- **Human-in-the-Loop (HITL)**: Manual correction dropdown for mispredictions
- **Compare Mode**: Side-by-side comparison of two foods
- **Demo Mode**: Test without trained model using 5 pre-loaded examples
- **Responsive Design**: Bootstrap 5 with mobile-first approach

---

## 🔧 Limitations

### Model Constraints
- **Limited Food Classes**: Only 18 Indian food items supported (see list below)
- **Domain Shift**: Model trained on curated dataset; real-world Google images may have lower confidence (40-60% → 75-90% after TTA)
- **Single Food Detection**: Cannot handle multiple foods in one image
- **No Portion Estimation**: Assumes standard serving size; user must manually adjust portion multiplier

### Environmental Requirements
- **Lighting Sensitivity**: Low light or extreme shadows reduce confidence
- **Angle Dependency**: Top-down or 45° angles work best; side views may fail
- **Background Noise**: Cluttered backgrounds can confuse the model

### GenAI Dependencies
- **Internet Required**: Groq API calls need active internet connection
- **API Key Required**: GenAI features disabled without valid GROQ_API_KEY
- **Rate Limits**: Free tier has request limits; may fallback to static content

---

## 🖼️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER (Web Browser)                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ Uploads Image
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FLASK APP (Controller)                             │
│                         app.py (Port 5001)                               │
└──────┬──────────────────────┬──────────────────────┬────────────────────┘
       │                      │                      │
       │ 1. Preprocess        │ 2. Get Nutrition     │ 3. Get GenAI
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Preprocessing│      │  JSON DB     │      │  Groq API    │
│  (Resize +   │      │ food_data.   │      │  (LLaMA-3.1  │
│  Normalize)  │      │  json        │      │   70b)       │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       │ 224×224 RGB         │ Calories, Macros    │ Cultural Context
       ▼                     │ Burn-off, Eco       │ Recipe Ideas
┌──────────────┐             │                     │
│ EfficientNet │             │                     │
│     B1       │             │                     │
│  (PyTorch)   │             │                     │
└──────┬───────┘             │                     │
       │                     │                     │
       │ Class ID            │                     │
       │ (e.g., "samosa")    │                     │
       └─────────┬───────────┴─────────────────────┘
                 │
                 │ Combine Results
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    JINJA2 TEMPLATE ENGINE                                │
│                  (Render HTML with Data)                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTML + CSS + JS
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    USER SCREEN (Bootstrap 5 UI)                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐             │
│  │  Overview   │ Nutrition   │   Chef's    │ Eco-Impact  │             │
│  │     Tab     │  Lab Tab    │  Kitchen    │     Tab     │             │
│  └─────────────┴─────────────┴─────────────┴─────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

**System follows a Model-View-Controller (MVC) pattern adaptation:**
- **Model**: EfficientNet-B1 (Deep Learning Logic) + FoodDataManager (Data Logic)
- **View**: Jinja2 Templates + Bootstrap 5 (Frontend Presentation)
- **Controller**: Flask Routes (app.py) mediating between Model and View

This ensures modularity; the model can be swapped (e.g., EfficientNet-B1 → ResNet50) without breaking the UI, and the UI can be redesigned without touching the ML pipeline.

---

## ⚡ Usage Notes

### How to Upload Images
1. Navigate to **Scan Food** page (`/scan`)
2. Click **"Choose File"** or drag-and-drop an image
3. Supported formats: JPG, JPEG, PNG
4. Click **"Analyze Food"** to submit

### Understanding the 4 Tabs

**1. Overview Tab**
- Quick stats: Calories, Eco Score, Protein
- Burn-Off Activities card (running, cycling, yoga)
- Portion slider (0.5× to 3.0×)
- Time-aware contextual intelligence (breakfast/lunch/dinner suggestions)

**2. Nutrition Lab Tab**
- Detailed nutrition table with % Daily Value
- Interactive macro radar chart (protein, carbs, fats)
- Macro sliders for "what-if" analysis
- Portion scaling controls

**3. Chef's Kitchen Tab**
- Cultural explanation (Ayurvedic properties, regional origins)
- Leftover ideas (creative repurposing suggestions)
- Flavor pairing (beverage and side dish recommendations)
- AI insights (3-bullet summary)

**4. Eco-Impact Tab**
- Sustainability score (1-10)
- Carbon footprint, water usage, food miles
- Eco-badge (green/yellow/red)
- AI-generated sustainability explanations

### Interpreting Confidence Scores

- **≥ 80%**: High confidence (green badge) - prediction is reliable
- **60-79%**: Medium confidence (yellow badge) - prediction is likely correct
- **< 60%**: Low confidence (red badge) - use HITL correction dropdown

**Why low confidence?**
- Domain shift: Real-world images differ from training data
- Poor lighting or extreme angles
- Cluttered background or multiple foods
- Food not in 18 supported classes

### Interactive Features

- **Portion Slider**: Drag to adjust serving size; all values update in real-time
- **Compare Mode**: Click "Scan Another Food" to compare two foods side-by-side
- **HITL Correction**: If prediction is wrong, select correct food from dropdown
- **Eco-Badge Animation**: Hover over badge for tooltip with sustainability details

---

## 🍛 Supported Foods (18 Classes)

1. **Samosa** - Triangular fried pastry with potato filling
2. **Idli** - Steamed rice cake (South Indian)
3. **Masala Dosa** - Crispy rice crepe with spiced potato filling
4. **Chole Bhature** - Spiced chickpeas with fried bread
5. **Pav Bhaji** - Spiced vegetable mash with bread rolls
6. **Paani Puri** - Crispy hollow puri with spiced water
7. **Butter Naan** - Leavened flatbread with butter
8. **Kadai Paneer** - Cottage cheese in spicy tomato gravy
9. **Dhokla** - Steamed fermented rice cake (Gujarati)
10. **Pakode** - Deep-fried fritters
11. **Momos** - Steamed dumplings (Tibetan-Indian)
12. **Kaathi Rolls** - Wrap with spiced filling
13. **Fried Rice** - Stir-fried rice with vegetables
14. **Jalebi** - Deep-fried sweet pretzel
15. **Kulfi** - Traditional Indian ice cream
16. **Chai** - Spiced milk tea
17. **Burger** - Western fast food (Indian adaptation)
18. **Pizza** - Italian dish (Indian adaptation)

---

## 📜 Attribution & Credits

### Frontend Template
- **Restoran Bootstrap Template** by [HTML Codex](https://htmlcodex.com/)
- Licensed under: Free to use with attribution

### Dataset
- **Indian Food 101 Dataset** (subset of 18 classes)
- Source: Kaggle + custom collection
- Total images: ~3,600 (200 per class)

### Libraries & Frameworks

**Deep Learning:**
- PyTorch 2.0+ (Model training and inference)
- torchvision (Pretrained EfficientNet-B1)
- Albumentations (Image augmentation)

**Web Framework:**
- Flask 2.3+ (Backend server)
- Jinja2 (Template engine)
- Flask-CORS (Cross-origin requests)

**Frontend:**
- Bootstrap 5 (CSS framework)
- Chart.js 4.4.0 (Radar chart visualization)
- Font Awesome (Icons)
- WOW.js (Scroll animations)

**GenAI:**
- Groq API (LLaMA-3.1-70b-versatile)
- python-dotenv (Environment variables)

**Image Processing:**
- Pillow 9.5+ (Image I/O)
- OpenCV 4.7+ (Image manipulation)

**Data Science:**
- NumPy 1.24+ (Numerical operations)
- pandas 2.0+ (Data manipulation)
- scikit-learn 1.2+ (Metrics and evaluation)

---

## 💡 Optional Improvements

### Deployment
- **Cloud Hosting**: Deploy to AWS EC2, Google Cloud Run, or Heroku
- **Containerization**: Create Dockerfile for reproducible deployment
- **HTTPS**: Add SSL certificate for secure connections
- **CDN**: Use Cloudflare or AWS CloudFront for static assets

### Feature Enhancements
- **User Accounts**: Add authentication and save food history
- **Barcode Scanner**: Integrate barcode reading for packaged foods
- **Multi-Food Detection**: Use object detection (YOLO) to handle multiple foods
- **Portion Estimation**: Use depth estimation or reference objects for automatic portion sizing
- **Meal Planning**: Suggest balanced meal combinations
- **Export Reports**: Generate PDF nutrition reports

### Model Improvements
- **Expand Classes**: Train on 50+ Indian food items
- **Fine-Grained Classification**: Distinguish between variations (e.g., plain dosa vs masala dosa)
- **Transfer Learning**: Use Vision Transformers (ViT) or ConvNeXt for better accuracy
- **Active Learning**: Collect user corrections to improve model over time

### Data Enhancements
- **Real-Time Nutrition API**: Integrate USDA FoodData Central or Nutritionix API
- **Regional Variations**: Add nutrition data for different regional preparations
- **Allergen Detection**: Flag common allergens (gluten, dairy, nuts)

---

## 👨‍💻 Author

**MSc ADSA Final Project**  
Data Analytics 3 - 3rd Semester  
December 2025

---

## 📄 License

**Educational Use Only**

This project is developed for academic purposes as part of a university final project. The code is provided as-is for educational reference.

**Third-Party Licenses:**
- Bootstrap 5: MIT License
- Chart.js: MIT License
- PyTorch: BSD License
- Flask: BSD License

---

## 🙏 Acknowledgments

Special thanks to:
- **HTML Codex** for the Restoran Bootstrap template
- **Kaggle community** for the Indian Food dataset
- **Groq** for providing free API access to LLaMA models
- **PyTorch team** for the excellent deep learning framework
- **University faculty** for guidance and support

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
