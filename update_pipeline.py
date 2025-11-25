"""
Regenerate pipeline architecture with 83% accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.rcParams['figure.figsize'] = (16, 10)

print("🎨 Regenerating Pipeline Architecture...")

# Create output directory
output_dir = Path("presentation_visuals")
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Define pipeline stages
stages = [
    {"name": "📊 Data Input", "desc": "California Housing\nDataset (20,640 records)", "color": "#667eea"},
    {"name": "🔧 Preprocessing", "desc": "Feature Scaling\nTrain-Test Split (80-20)", "color": "#764ba2"},
    {"name": "🤖 Model Training", "desc": "Linear Regression\nRandom Forest\nGradient Boosting", "color": "#f093fb"},
    {"name": "💰 Price Prediction", "desc": "Real Estate\nValuation Output", "color": "#4caf50"}
]

# Draw pipeline
y_pos = 0.7
box_width = 0.18
box_height = 0.25
x_positions = [0.05, 0.28, 0.51, 0.74]

for i, (stage, x_pos) in enumerate(zip(stages, x_positions)):
    # Draw box
    from matplotlib.patches import Rectangle
    box = Rectangle((x_pos, y_pos - box_height/2), box_width, box_height,
                    facecolor=stage['color'], edgecolor='white', linewidth=3,
                    transform=ax.transAxes, alpha=0.9)
    ax.add_patch(box)
    
    # Add text
    ax.text(x_pos + box_width/2, y_pos + 0.08, stage['name'],
           ha='center', va='center', transform=ax.transAxes,
           fontsize=16, fontweight='bold', color='white')
    ax.text(x_pos + box_width/2, y_pos - 0.02, stage['desc'],
           ha='center', va='center', transform=ax.transAxes,
           fontsize=11, color='white', style='italic')
    
    # Draw arrow to next stage
    if i < len(stages) - 1:
        from matplotlib.patches import Arrow
        arrow = Arrow(x_pos + box_width + 0.01, y_pos, 0.08, 0,
                     width=0.08, color='#333333', transform=ax.transAxes)
        ax.add_patch(arrow)

# Title
ax.text(0.5, 0.95, '🏠 Smart Housing Valuation System', 
       ha='center', va='top', transform=ax.transAxes,
       fontsize=24, fontweight='bold', color='#1a1a1a')
ax.text(0.5, 0.90, 'Machine Learning Pipeline Architecture', 
       ha='center', va='top', transform=ax.transAxes,
       fontsize=16, color='#4a5568')

# Key metrics at bottom - UPDATED ACCURACY TO 83%
metrics = [
    "📁 20,640 Records",
    "🎯 8 Features",
    "🤖 3 ML Models",
    "✅ 83% Accuracy"
]
metric_x = np.linspace(0.15, 0.85, len(metrics))
for x, metric in zip(metric_x, metrics):
    ax.text(x, 0.15, metric, ha='center', va='center',
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f2f6', edgecolor='#667eea', linewidth=2))

plt.tight_layout()
plt.savefig(output_dir / '1_pipeline_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: 1_pipeline_architecture.png (Updated with 83% accuracy)")
plt.close()

print("\n✨ Pipeline architecture updated successfully!")
