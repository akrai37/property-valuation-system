"""
Code Examples & Advanced Usage Guide
Smart Housing Valuation System
"""

# ============================================================================
# EXAMPLE 1: Make a Single Prediction Programmatically
# ============================================================================

from predict import PricePredictorUtil

# Initialize the predictor (loads saved model and scaler)
predictor = PricePredictorUtil()

# Make a prediction for a specific property
result = predictor.predict(
    MedInc=6.5,           # $65,000 median income
    HouseAge=15,          # 15 years old
    AveRooms=7.0,         # 7 rooms average
    AveBedrms=3.5,        # 3.5 bedrooms average
    Population=5000,      # 5000 people in block
    AveOccup=3.2,         # 3.2 people per household
    Latitude=37.8,        # Northern California
    Longitude=-122.4      # San Francisco Bay area
)

# Access the results
print(f"Predicted Price: ${result['predicted_price']:,.2f}")
print(f"Raw Value: {result['prediction_value']:.4f}")
print(f"Input Features: {result['input_features']}")


# ============================================================================
# EXAMPLE 2: Batch Predictions from DataFrame
# ============================================================================

import pandas as pd
from predict import PricePredictorUtil

# Create a DataFrame with multiple properties
properties = pd.DataFrame({
    'MedInc': [4.0, 6.5, 3.5, 8.0],
    'HouseAge': [20, 10, 45, 5],
    'AveRooms': [5.0, 7.0, 4.0, 8.0],
    'AveBedrms': [2.0, 3.5, 1.5, 4.0],
    'Population': [3000, 5000, 12000, 2000],
    'AveOccup': [3.0, 3.2, 4.5, 2.8],
    'Latitude': [37.0, 37.8, 38.0, 36.5],
    'Longitude': [-121.0, -122.4, -122.0, -120.5]
})

# Initialize predictor
predictor = PricePredictorUtil()

# Make batch predictions
predictions = predictor.predict_batch(properties)

# Add predictions to dataframe
properties['predicted_price'] = predictions

# View results
print(properties[['MedInc', 'HouseAge', 'predicted_price']])


# ============================================================================
# EXAMPLE 3: Custom Model Training with Different Data
# ============================================================================

import pandas as pd
from train_model import HousingModelPipeline

# Load your own dataset instead
# custom_X = pd.read_csv('your_housing_data.csv')
# custom_y = custom_X['price']
# custom_X = custom_X.drop('price', axis=1)

# Or use the default California dataset
pipeline = HousingModelPipeline()
X, y = pipeline.load_data()

# Preprocess
X_train, X_test, y_train, y_test, features = pipeline.preprocess_data(X, y)

# Train only specific models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression R²: {lr.score(X_test, y_test):.4f}")

rf = RandomForestRegressor(n_estimators=200)  # More trees
rf.fit(X_train, y_train)
print(f"Random Forest R²: {rf.score(X_test, y_test):.4f}")


# ============================================================================
# EXAMPLE 4: Evaluate Models with Cross-Validation
# ============================================================================

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from train_model import HousingModelPipeline

# Load and preprocess data
pipeline = HousingModelPipeline()
X, y = pipeline.load_data()
X_train, X_test, y_train, y_test, _ = pipeline.preprocess_data(X, y)

# Cross-validation for Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f"Random Forest CV Scores: {cv_scores_rf}")
print(f"Mean R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# Cross-validation for Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
cv_scores_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring='r2')
print(f"Gradient Boosting CV Scores: {cv_scores_gb}")
print(f"Mean R²: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")


# ============================================================================
# EXAMPLE 5: Hyperparameter Tuning with GridSearchCV
# ============================================================================

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from train_model import HousingModelPipeline

# Load and preprocess data
pipeline = HousingModelPipeline()
X, y = pipeline.load_data()
X_train, X_test, y_train, y_test, _ = pipeline.preprocess_data(X, y)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Create GridSearchCV
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)

# Fit and find best parameters
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Train with best parameters
best_rf = grid_search.best_estimator_
print(f"Test Score: {best_rf.score(X_test, y_test):.4f}")


# ============================================================================
# EXAMPLE 6: Feature Importance Analysis
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from train_model import HousingModelPipeline

# Load and preprocess data
pipeline = HousingModelPipeline()
X, y = pipeline.load_data()
X_train, X_test, y_train, y_test, features = pipeline.preprocess_data(X, y)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Create comprehensive feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (model, name) in enumerate([(rf, 'Random Forest'), 
                                       (gb, 'Gradient Boosting')]):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax = axes[idx]
    ax.bar(range(len(importances)), importances[indices], color='steelblue')
    ax.set_xlabel('Features', fontsize=11)
    ax.set_ylabel('Importance', fontsize=11)
    ax.set_title(f'{name} Feature Importance', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(importances[indices]):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# ============================================================================
# EXAMPLE 7: Create Custom Streamlit Components
# ============================================================================

import streamlit as st
import pandas as pd
from predict import PricePredictorUtil

# Custom metric display
def display_prediction_metrics(result):
    """Display prediction results in custom format."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Price",
            value=f"${result['predicted_price']:,.0f}",
            delta=None,
        )
    
    with col2:
        st.metric(
            label="Model Confidence",
            value="High",
            delta="+0.7 R²"
        )
    
    with col3:
        st.metric(
            label="Estimated Range",
            value=f"±${result['predicted_price']*0.15:,.0f}"
        )

# Custom prediction form
def get_property_inputs():
    """Get property inputs from user."""
    with st.form("property_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            medinc = st.number_input("Median Income", 0.5, 15.0, 5.0)
            house_age = st.number_input("House Age", 1, 52, 25)
            ave_rooms = st.number_input("Average Rooms", 1.0, 10.0, 5.5)
            ave_bedrms = st.number_input("Average Bedrooms", 0.5, 5.0, 2.0)
        
        with col2:
            population = st.number_input("Population", 100, 35000, 5000)
            ave_occup = st.number_input("Average Occupancy", 1.0, 10.0, 3.0)
            latitude = st.number_input("Latitude", 32.0, 42.0, 37.5)
            longitude = st.number_input("Longitude", -125.0, -114.0, -120.0)
        
        submitted = st.form_submit_button("Predict Price")
        
        if submitted:
            return {
                'MedInc': medinc,
                'HouseAge': house_age,
                'AveRooms': ave_rooms,
                'AveBedrms': ave_bedrms,
                'Population': population,
                'AveOccup': ave_occup,
                'Latitude': latitude,
                'Longitude': longitude
            }
    return None


# ============================================================================
# EXAMPLE 8: Performance Comparison Visualization
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from train_model import HousingModelPipeline

# Load and preprocess data
pipeline = HousingModelPipeline()
X, y = pipeline.load_data()
X_train, X_test, y_train, y_test, _ = pipeline.preprocess_data(X, y)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['MSE', 'MAE', 'R²']
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    comparison_df[metric].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(comparison_df)


# ============================================================================
# EXAMPLE 9: Error Analysis
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from train_model import HousingModelPipeline

# Load and preprocess data
pipeline = HousingModelPipeline()
X, y = pipeline.load_data()
X_train, X_test, y_train, y_test, _ = pipeline.preprocess_data(X, y)

# Train best model
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

# Calculate errors
residuals = y_test - y_pred
absolute_errors = np.abs(residuals)

# Create error analysis plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Predicted vs Actual
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.set_title('Predicted vs Actual')
ax.grid(True, alpha=0.3)

# 2. Residuals
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Price')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

# 3. Absolute Error Distribution
ax = axes[1, 0]
ax.hist(absolute_errors, bins=50, color='steelblue', edgecolor='black')
ax.set_xlabel('Absolute Error')
ax.set_ylabel('Frequency')
ax.set_title(f'Error Distribution (Mean: {absolute_errors.mean():.4f})')
ax.grid(True, alpha=0.3, axis='y')

# 4. Error Percentage
ax = axes[1, 1]
error_percentage = (absolute_errors / y_test.values) * 100
ax.hist(error_percentage, bins=50, color='orange', edgecolor='black')
ax.set_xlabel('Error Percentage (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'Error Percentage Distribution (Mean: {error_percentage.mean():.2f}%)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# ============================================================================
# EXAMPLE 10: Deploy as API with Flask (Optional)
# ============================================================================

"""
from flask import Flask, request, jsonify
from predict import PricePredictorUtil
import json

app = Flask(__name__)

# Load predictor on startup
predictor = PricePredictorUtil()

@app.route('/predict', methods=['POST'])
def predict():
    '''API endpoint for price prediction'''
    try:
        data = request.json
        result = predictor.predict(**data)
        return jsonify({
            'status': 'success',
            'predicted_price': result['predicted_price'],
            'prediction_value': result['prediction_value']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    '''API endpoint for batch predictions'''
    try:
        import pandas as pd
        data = request.json
        df = pd.DataFrame(data)
        predictions = predictor.predict_batch(df)
        return jsonify({
            'status': 'success',
            'predictions': predictions.tolist()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Usage:
# curl -X POST http://localhost:5000/predict \\
#   -H "Content-Type: application/json" \\
#   -d '{"MedInc": 5.0, "HouseAge": 25, ...}'
"""

print(__doc__)
