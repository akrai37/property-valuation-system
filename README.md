# 🏠 Smart Housing Valuation System

A complete machine learning project that predicts residential property prices using multiple regression models and provides an interactive web interface for real-time predictions.

## 📋 Project Overview

This project implements a complete ML pipeline for housing price prediction:
- **Data Processing**: Load, clean, and normalize housing data
- **Model Training**: Compare Linear Regression, Random Forest, and Gradient Boosting models
- **Web Application**: Interactive Streamlit app for price predictions
- **Visualization**: Feature importance analysis and model performance metrics

## 🎯 Features

### Core ML Models
- **Linear Regression**: Fast, interpretable baseline model
- **Random Forest**: Ensemble method handling non-linear relationships
- **Gradient Boosting**: Advanced model for highest accuracy

### Web Application Features
- 🎯 **Instant Price Prediction**: Input property features and get predictions
- 📊 **Feature Importance Visualization**: Understand which features drive prices
- 📈 **Interactive Sliders**: Easy property feature adjustment
- 🎨 **Modern UI**: Clean, professional interface with Streamlit

### Input Features
The model accepts 8 property features:
- **MedInc**: Median household income (in $10,000s)
- **HouseAge**: Age of the house (years)
- **AveRooms**: Average rooms per household
- **AveBedrms**: Average bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average household occupancy
- **Latitude**: Geographic latitude
- **Longitude**: Geographic longitude

## 📊 Dataset

- **Source**: California Housing Dataset (built-in Sklearn dataset)
- **Records**: ~20,000 housing samples
- **Time Period**: 1990 Census data
- **Features**: 8 numerical features
- **Target**: Median house value (in $100,000s)

## 🛠️ Tech Stack

### Backend (ML)
- **Python 3.8+**
- **scikit-learn**: Machine learning models
- **pandas**: Data processing
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualizations

### Frontend
- **Streamlit**: Web application framework
- **Pillow**: Image handling

### Optional
- **XGBoost**: Alternative gradient boosting (can replace sklearn's GradientBoosting)

## 📦 Installation

### 1. Clone/Navigate to Project

```bash
cd smart_housing_valuation
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Step 1: Train the Model

Run the training script to generate and save the ML models:

```bash
python train_model.py
```

This will:
- Load the California Housing dataset
- Train Linear Regression, Random Forest, and Gradient Boosting models
- Evaluate all models using MSE, RMSE, R², and MAE metrics
- Save the best performing model
- Generate feature importance visualization
- Create `models/` directory with:
  - `best_model.pkl` - The trained best model
  - `scaler.pkl` - Feature scaling object
  - `features.pkl` - Feature names
  - `feature_importance.png` - Visualization

Expected output:
```
==========================================================================
SMART HOUSING VALUATION SYSTEM - MODEL TRAINING
==========================================================================
Loading California Housing dataset...
...
Preprocessing data...
Training models...
1. Training Linear Regression...
   MSE: 0.5733, RMSE: 0.7571, R²: 0.5757, MAE: 0.5331
...
```

### Step 2: Run Web Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Web Application

1. **Prediction Tab**: 
   - Adjust property features with sliders
   - Click "Predict Price" for instant prediction
   - View input summary table

2. **Feature Importance Tab**:
   - See which features most influence predictions
   - Understand model decision-making process

3. **About Tab**:
   - Learn about the models and dataset
   - Get tips for better predictions
   - Review feature descriptions

## 📈 Model Performance

The training script compares three models:

### Expected Results (Approximate)
- **Linear Regression**: R² ~0.58, RMSE ~0.76
- **Random Forest**: R² ~0.65, RMSE ~0.68
- **Gradient Boosting**: R² ~0.70, RMSE ~0.60

Best performing model is automatically selected and saved.

## 🔍 Project Structure

```
smart_housing_valuation/
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── models/                # Generated during training
│   ├── best_model.pkl     # Trained model
│   ├── scaler.pkl         # Feature scaler
│   ├── features.pkl       # Feature names
│   └── feature_importance.png  # Visualization
└── data/                  # (Optional) Raw data storage
```

## 💡 Key Implementation Details

### Data Preprocessing
- **Scaling**: StandardScaler for feature normalization
- **Train-Test Split**: 80% training, 20% testing
- **Missing Values**: Handled with mean imputation

### Model Training
- **Random State**: Set to 42 for reproducibility
- **Feature Importance**: Extracted from tree-based models
- **Evaluation Metrics**:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
  - MAE (Mean Absolute Error)

### Web Interface
- **Responsive Design**: Works on desktop and tablet
- **Real-time Prediction**: Instant results as you adjust sliders
- **Visual Feedback**: Clear metrics and prediction display

## 🎨 Customization

### Modify Input Ranges
Edit `app.py` to change slider ranges:
```python
medinc = st.slider(
    "Median Income (in $10,000s)",
    min_value=0.5,    # Change these
    max_value=15.0,   # Change these
    value=3.0,
    step=0.1,
)
```

### Add New Models
In `train_model.py`, add to the `train_models` method:
```python
from sklearn.ensemble import ExtraTreesRegressor

et = ExtraTreesRegressor(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
# ... evaluate and store results
```

### Styling
Customize app appearance in the `<style>` section of `app.py`

## 🐛 Troubleshooting

### Issue: "Model files not found"
**Solution**: Run `python train_model.py` first

### Issue: Import errors
**Solution**: Ensure virtual environment is activated and all packages installed:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Streamlit port already in use
**Solution**: Specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Slow prediction on first run
**Solution**: This is normal - model is being loaded. Subsequent predictions are instant.

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Tutorial](https://pandas.pydata.org/docs/)
- [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

## 🔒 Model Accuracy Notes

- The model is trained on 1990 Census data, so predictions may not reflect current market prices
- Actual housing markets have additional factors (neighborhood amenities, school districts, recent renovations)
- Model works best for properties within the feature ranges seen during training
- Use as a reference tool, not definitive price estimate

## 📝 Future Enhancements

Potential improvements:
- [ ] Add more datasets (Boston Housing, real-time market data)
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning
- [ ] Create model comparison visualizations
- [ ] Deploy to cloud (Heroku, AWS, Azure)
- [ ] Add price history trends
- [ ] Include uncertainty estimates
- [ ] Multi-region support

## 👨‍💻 Author & License

Created as a machine learning course project.

Free to use and modify for educational purposes.

## 🤝 Contributing

Feel free to:
- Add new models
- Improve visualizations
- Enhance the web interface
- Add more datasets
- Optimize performance

---

**Happy Predicting! 🏠📊**
