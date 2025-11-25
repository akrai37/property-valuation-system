"""
Regenerate feature importance chart with Gradient Boosting label
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

print("📊 Regenerating Feature Importance Chart...")

# Load model
models_dir = Path("models")
with open(models_dir / "best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data to get feature names
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

# Create output directory
output_dir = Path("presentation_visuals")
output_dir.mkdir(exist_ok=True)

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 10))

# Create gradient colors
colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))

bars = ax.barh(feature_importance['Feature'], 
               feature_importance['Importance'] * 100,
               color=colors_gradient, edgecolor='black', linewidth=1.5)

# Highlight MedInc (should be the longest bar)
max_idx = feature_importance['Importance'].idxmax()
bars[list(feature_importance.index).index(max_idx)].set_color('#4caf50')
bars[list(feature_importance.index).index(max_idx)].set_edgecolor('gold')
bars[list(feature_importance.index).index(max_idx)].set_linewidth(3)

# Add percentage labels
for i, (idx, row) in enumerate(feature_importance.iterrows()):
    ax.text(row['Importance'] * 100 + 1, i, f"{row['Importance']*100:.1f}%",
           va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Importance (%)', fontweight='bold', fontsize=14)
ax.set_ylabel('Feature', fontweight='bold', fontsize=14)
ax.set_title('📊 Feature Importance Analysis\nGradient Boosting Model - Impact on House Price Predictions', 
            fontweight='bold', fontsize=18, pad=20)
ax.grid(True, alpha=0.3, axis='x')

# Feature name mapping
feature_descriptions = {
    'MedInc': '💰 Median Income',
    'HouseAge': '📅 House Age',
    'AveRooms': '🚪 Avg Rooms',
    'AveBedrms': '🛏️ Avg Bedrooms',
    'Population': '👥 Population',
    'AveOccup': '👨‍👩‍👧‍👦 Avg Occupancy',
    'Latitude': '🧭 Latitude',
    'Longitude': '🧭 Longitude'
}

# Update y-tick labels
current_labels = [label.get_text() for label in ax.get_yticklabels()]
new_labels = [feature_descriptions.get(label, label) for label in current_labels]
ax.set_yticklabels(new_labels)

plt.tight_layout()
plt.savefig(output_dir / '6_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 6_feature_importance.png (Updated with Gradient Boosting label)")
plt.close()

print("\n✨ Feature importance chart updated successfully!")
