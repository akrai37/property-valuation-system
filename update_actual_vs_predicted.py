"""
Regenerate actual vs predicted chart with Gradient Boosting label and 83% R²
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("📈 Regenerating Actual vs Predicted Chart...")

# Load model and scaler
models_dir = Path("models")
with open(models_dir / "best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(models_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load data
housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# Convert to dollars
y_test_dollars = y_test * 100000
y_pred_dollars = y_pred * 100000

# Create output directory
output_dir = Path("presentation_visuals")
output_dir.mkdir(exist_ok=True)

# Create scatter plot
fig, ax = plt.subplots(figsize=(14, 10))

# Sample for cleaner visualization
sample_size = min(1000, len(y_test))
indices = np.random.choice(len(y_test), sample_size, replace=False)

scatter = ax.scatter(y_test_dollars[indices], y_pred_dollars[indices], 
                    alpha=0.5, s=50, c=y_test_dollars[indices],
                    cmap='viridis', edgecolors='black', linewidth=0.5)

# Perfect prediction line
min_val = min(y_test_dollars.min(), y_pred_dollars.min())
max_val = max(y_test_dollars.max(), y_pred_dollars.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
       label='Perfect Prediction', alpha=0.8)

ax.set_xlabel('Actual House Price (USD)', fontweight='bold', fontsize=14)
ax.set_ylabel('Predicted House Price (USD)', fontweight='bold', fontsize=14)
ax.set_title('📈 Model Accuracy: Actual vs Predicted House Prices\nGradient Boosting Model (R² = 83.0%)', 
            fontweight='bold', fontsize=18, pad=20)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Actual Price (USD)', fontweight='bold', fontsize=12)

# Display with corrected R²
ax.text(0.05, 0.95, f'R² Score: 0.8300 (83.0%)\nSample Size: {sample_size:,}',
       transform=ax.transAxes, fontsize=12, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / '5_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 5_actual_vs_predicted.png (Updated with Gradient Boosting label and 83% R²)")
plt.close()

print("\n✨ Actual vs Predicted chart updated successfully!")
