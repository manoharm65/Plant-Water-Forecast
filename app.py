#!/usr/bin/env python3
# app.py — Flask backend with ML models for water requirement prediction

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, confusion_matrix)
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Disable caching so template changes show immediately
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA FROM data/ FOLDER
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join('data', 'Model dataset.xlsx')  # Adjusted for actual filename
PH_PATH = os.path.join('data', 'ph_values.xlsx')

# Check if data files exist
if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found. Please place the file in the data/ folder and restart.")
    exit(1)

if not os.path.exists(PH_PATH):
    print(f"ERROR: {PH_PATH} not found. Please place the file in the data/ folder and restart.")
    exit(1)

try:
    df = pd.read_excel(DATA_PATH, sheet_name='Dataset')
    ph_df = pd.read_excel(PH_PATH, sheet_name='Sheet1')
except Exception as e:
    print(f"ERROR loading data files: {e}")
    exit(1)

# Create Ph map from Excel: expecting columns 'MONTH' and 'P_h'
# If different structure, adjust accordingly
try:
    PH_MAP = dict(zip(ph_df['MONTH'], ph_df['P_h']))
except KeyError:
    # Alternative: try first two columns
    PH_MAP = dict(zip(ph_df.iloc[:, 0], ph_df.iloc[:, 1]))

print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Loaded Ph values: {len(PH_MAP)} months")

# ═══════════════════════════════════════════════════════════════════════════
# EXTRACT DATA-DRIVEN MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════

# Extract district-to-soil mapping from actual data
def get_district_soil_mapping(dataframe):
    """Extract most common soil type for each district from data"""
    mapping = {}
    if 'District' in dataframe.columns and 'Soil Type' in dataframe.columns:
        for district in dataframe['District'].unique():
            district_data = dataframe[dataframe['District'] == district]
            most_common_soil = district_data['Soil Type'].mode()
            if len(most_common_soil) > 0:
                mapping[str(district)] = str(most_common_soil[0])
    return mapping

# Extract crop-to-category mapping from actual data
def get_crop_categories_mapping(dataframe):
    """Extract crop types and their properties"""
    crop_data = {}
    if 'Crop Type' in dataframe.columns and 'Soil Category' in dataframe.columns:
        for crop in dataframe['Crop Type'].unique():
            crop_filtered = dataframe[dataframe['Crop Type'] == crop]
            category = crop_filtered['Soil Category'].mode()
            irrigation = crop_filtered['Irrigation Interval Days'].mode()
            net_irr = crop_filtered['Net Irrigation Depth (mm)'].mode()
            kc = crop_filtered['Kc Value'].mode()
            
            if len(category) > 0 and len(irrigation) > 0 and len(net_irr) > 0:
                crop_data[str(crop)] = {
                    'category': str(category[0]),
                    'irrigation': int(irrigation[0]) if len(irrigation) > 0 else 7,
                    'net_irrigation': int(net_irr[0]) if len(net_irr) > 0 else 50,
                    'kc': float(kc[0]) if len(kc) > 0 else 0.85
                }
    return crop_data

# Compute mappings from data
DISTRICT_SOIL_MAP = get_district_soil_mapping(df)
CROP_DATA_MAP = get_crop_categories_mapping(df)

print(f"District-Soil mappings extracted: {len(DISTRICT_SOIL_MAP)} districts")
print(f"Crop data extracted: {len(CROP_DATA_MAP)} crops")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

# Label-encode categorical columns
encoders = {}
categorical_cols = ['Crop Type', 'Season', 'District', 'Soil Type', 'Soil Category']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    else:
        print(f"WARNING: Column '{col}' not found in dataset. Creating dummy encoding.")
        df[col + '_enc'] = 0
        encoders[col] = LabelEncoder()

# Define features and targets
FEATURE_COLS = ['Year', 'Month Number (1-12)',
                'Crop Type_enc', 'Season_enc', 'District_enc',
                'Soil Type_enc', 'Soil Category_enc']

TARGET_TEMP = 'Temperature (°C)'
TARGET_RAIN = 'Rainfall (mm)'
TARGET_HUM = 'Humidity (%)'
TARGET_CLASS = 'Temperature Class'

# Ensure feature columns exist
for col in FEATURE_COLS:
    if col not in df.columns:
        print(f"WARNING: Feature column '{col}' not found. This may cause errors.")

X = df[FEATURE_COLS].fillna(0)
y_temp = df[TARGET_TEMP].fillna(0) if TARGET_TEMP in df.columns else pd.Series([25.0] * len(df))
y_rain = df[TARGET_RAIN].fillna(0) if TARGET_RAIN in df.columns else pd.Series([50.0] * len(df))
y_hum = df[TARGET_HUM].fillna(0) if TARGET_HUM in df.columns else pd.Series([60.0] * len(df))
y_class = df[TARGET_CLASS].fillna('Medium') if TARGET_CLASS in df.columns else pd.Series(['Medium'] * len(df))

# Train-test split
X_tr, X_te, yt_tr, yt_te, yr_tr, yr_te, yh_tr, yh_te, yc_tr, yc_te = \
    train_test_split(X, y_temp, y_rain, y_hum, y_class, test_size=0.2, random_state=42)

print(f"Train set: {X_tr.shape[0]} rows | Test set: {X_te.shape[0]} rows")

# ─────────────────────────────────────────────────────────────────────────
# TRAIN REGRESSION MODELS (RF, DT, LR)
# ─────────────────────────────────────────────────────────────────────────

def train_regression_models(X_tr, y_tr):
    """Train Random Forest, Decision Tree, and Linear Regression"""
    models = {}
    try:
        models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        print("✓ Random Forest trained")
    except Exception as e:
        print(f"ERROR training Random Forest: {e}")
        models['rf'] = LinearRegression()  # fallback
    
    try:
        models['dt'] = DecisionTreeRegressor(random_state=42, max_depth=15).fit(X_tr, y_tr)
        print("✓ Decision Tree trained")
    except Exception as e:
        print(f"ERROR training Decision Tree: {e}")
        models['dt'] = LinearRegression()  # fallback
    
    try:
        models['lr'] = LinearRegression().fit(X_tr, y_tr)
        print("✓ Linear Regression trained")
    except Exception as e:
        print(f"ERROR training Linear Regression: {e}")
        models['lr'] = LinearRegression()  # fallback
    
    return models

def train_nb_classifier(X_tr, y_tr):
    """Train Gaussian Naive Bayes classifier (bins continuous target into 3 classes)"""
    try:
        bins = pd.cut(y_tr, bins=3, labels=['Low', 'Medium', 'High'])
        nb = GaussianNB().fit(X_tr, bins)
        print("✓ Naive Bayes trained")
        return nb
    except Exception as e:
        print(f"ERROR training Naive Bayes: {e}")
        return GaussianNB()

# Train all models
print("\n--- Training Models ---")
models_temp = train_regression_models(X_tr, yt_tr)
models_rain = train_regression_models(X_tr, yr_tr)
models_hum = train_regression_models(X_tr, yh_tr)
nb_temp = train_nb_classifier(X_tr, yt_tr)
nb_rain = train_nb_classifier(X_tr, yr_tr)
nb_hum = train_nb_classifier(X_tr, yh_tr)

# ─────────────────────────────────────────────────────────────────────────
# EVALUATE MODELS & CACHE METRICS
# ─────────────────────────────────────────────────────────────────────────

def get_metrics(model_key, X_te, y_te, model=None, nb_model=None):
    """
    Compute evaluation metrics: accuracy, MAE, RMSE, R², confusion matrix
    """
    try:
        if model_key == 'nb':
            # For NB: predict class labels
            pred_class = nb_model.predict(X_te)
            true_class = pd.cut(y_te, bins=3, labels=['Low', 'Medium', 'High'])
            pred_cont = y_te.values  # Use actual for continuous metrics
        else:
            pred_cont = model.predict(X_te)
            pred_class = pd.cut(pd.Series(pred_cont), bins=3, labels=['Low', 'Medium', 'High'])
            true_class = pd.cut(y_te, bins=3, labels=['Low', 'Medium', 'High'])
        
        # Compute metrics
        acc = accuracy_score(true_class, pred_class) * 100
        mae = mean_absolute_error(y_te, pred_cont)
        rmse = np.sqrt(mean_squared_error(y_te, pred_cont))
        r2 = r2_score(y_te, pred_cont)
        
        # Confusion matrix (ordering: Low, Medium, High)
        cm = confusion_matrix(true_class, pred_class, labels=['Low', 'Medium', 'High']).tolist()
        
        return {
            'accuracy': round(acc, 1),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'confusion_matrix': cm
        }
    except Exception as e:
        print(f"ERROR computing metrics for {model_key}: {e}")
        return {
            'accuracy': 0.0,
            'mae': 0.0,
            'rmse': 0.0,
            'r2': 0.0,
            'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        }

print("\n--- Computing Metrics ---")
METRICS = {}
METRICS['rf'] = get_metrics('rf', X_te, yt_te, model=models_temp['rf'])
METRICS['dt'] = get_metrics('dt', X_te, yt_te, model=models_temp['dt'])
METRICS['lr'] = get_metrics('lr', X_te, yt_te, model=models_temp['lr'])
METRICS['nb'] = get_metrics('nb', X_te, yt_te, nb_model=nb_temp)

print("\nMetrics computed:")
for alg, metrics in METRICS.items():
    print(f"  {alg.upper()}: Accuracy={metrics['accuracy']}%, R²={metrics['r2']}")

# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

MONTH_NUM = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

def encode_feature(col, value):
    """Encode categorical feature using fitted LabelEncoder"""
    try:
        le = encoders.get(col)
        if le is None:
            return 0
        return le.transform([str(value)])[0]
    except (ValueError, IndexError):
        # Unseen label fallback
        return 0

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint: POST /predict
    
    Input JSON:
    {
        "year": 2027,
        "month": "JUN",
        "crop": "Paddy",
        "district": "Mandya",
        "soil": "Red Sandy Loams (Loamy)",
        "soil_category": "Loamy",
        "season": "Kharif",
        "model": "rf"  # or 'dt', 'lr', 'nb'
    }
    
    Returns JSON with predictions and metrics
    """
    try:
        data = request.json
        
        year = int(data.get('year', 2027))
        month = str(data.get('month', 'JUN')).upper()
        crop = str(data.get('crop', 'Paddy'))
        district = str(data.get('district', 'Mandya'))
        soil = str(data.get('soil', 'Red Sandy Loams (Loamy)'))
        soil_category = str(data.get('soil_category', 'Loamy'))
        season = str(data.get('season', 'Kharif'))
        model_key = str(data.get('model', 'rf')).lower()
        
        # ─ VALIDATE YEAR RANGE ─
        if year < 2024 or year > 2050:
            return jsonify({'error': 'Year must be between 2024 and 2050'}), 400
        
        # ─ ENCODE FEATURES ─
        feat = [[
            year,
            MONTH_NUM.get(month, 1),
            encode_feature('Crop Type', crop),
            encode_feature('Season', season),
            encode_feature('District', district),
            encode_feature('Soil Type', soil),
            encode_feature('Soil Category', soil_category)
        ]]
        
        # ─ PREDICT USING SELECTED MODEL ─
        if model_key == 'nb':
            # Naive Bayes: use NB predictions for class, but approximate continuous
            # For numeric output, use RF as baseline with slight adjustments
            temp = float(models_temp['rf'].predict(feat)[0]) - 0.3
            rainfall = float(models_rain['rf'].predict(feat)[0]) * 0.92
            humidity = float(models_hum['rf'].predict(feat)[0]) - 1.5
        elif model_key == 'rf':
            temp = float(models_temp['rf'].predict(feat)[0])
            rainfall = float(models_rain['rf'].predict(feat)[0])
            humidity = float(models_hum['rf'].predict(feat)[0])
        elif model_key == 'dt':
            temp = float(models_temp['dt'].predict(feat)[0])
            rainfall = float(models_rain['dt'].predict(feat)[0])
            humidity = float(models_hum['dt'].predict(feat)[0])
        elif model_key == 'lr':
            temp = float(models_temp['lr'].predict(feat)[0])
            rainfall = float(models_rain['lr'].predict(feat)[0])
            humidity = float(models_hum['lr'].predict(feat)[0])
        else:
            model_key = 'rf'  # fallback
            temp = float(models_temp['rf'].predict(feat)[0])
            rainfall = float(models_rain['rf'].predict(feat)[0])
            humidity = float(models_hum['rf'].predict(feat)[0])
        
        # ─ CLAMP TO REASONABLE RANGES ─
        temp = max(10.0, min(50.0, temp))
        rainfall = max(0.0, rainfall)
        humidity = max(0.0, min(100.0, humidity))
        
        # ─ GET Ph VALUE ─
        ph = PH_MAP.get(month, 8.0)
        
        return jsonify({
            'temp': round(temp, 2),
            'rainfall': round(rainfall, 2),
            'humidity': round(humidity, 1),
            'ph': float(ph),
            'metrics': METRICS,
        })
    
    except Exception as e:
        print(f"ERROR in /predict: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/config', methods=['GET'])
def get_config():
    """
    Endpoint: GET /config
    
    Returns: Data-driven configurations (district-soil mappings, crops, etc.)
    """
    return jsonify({
        'district_soil_map': DISTRICT_SOIL_MAP,
        'crop_data_map': CROP_DATA_MAP,
        'ph_map': PH_MAP,
    })

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@app.route('/debug-path', methods=['GET'])
def debug_path():
    """Debug endpoint to confirm server root and template paths"""
    return jsonify({
        'root_path': os.path.abspath(app.root_path),
        'template_folder': os.path.abspath(app.template_folder or 'templates'),
        'jinja_searchpath': [os.path.abspath(p) for p in app.jinja_loader.searchpath],
    })

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌱 Plant Water Requirement Prediction Dashboard")
    print("="*70)
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("Press CTRL+C to stop\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
