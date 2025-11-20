"""
Check actual metrics from trained models
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load model and scaler
models_dir = Path("models")
with open(models_dir / "best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(models_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load data (same split as training)
housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("="*60)
print("ACTUAL METRICS FROM YOUR TRAINED MODEL")
print("="*60)
print(f"Model Type: {type(model).__name__}")
print(f"\nTest Set Performance:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f} ({r2*100:.2f}%)")
print(f"  MAE:  {mae:.4f}")
print("="*60)
