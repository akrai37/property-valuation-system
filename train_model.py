"""
Smart Housing Valuation System - Model Training Script
This script trains multiple regression models on housing price data
and saves the best performing model along with preprocessing artifacts.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HousingModelPipeline:
    """Pipeline for training and evaluating housing price prediction models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load California Housing dataset and prepare it."""
        print("Loading California Housing dataset...")
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target, name='price')
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"\nPrice statistics:")
        print(y.describe())
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Handle missing values and scale features."""
        print("\nPreprocessing data...")
        
        # Handle missing values (if any)
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        print(f"Training set size: {X_train_scaled.shape[0]}")
        print(f"Test set size: {X_test_scaled.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple regression models."""
        print("\nTraining models...")
        
        # Linear Regression
        print("\n1. Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        self.models['Linear Regression'] = lr
        self.results['Linear Regression'] = {
            'MSE': mse_lr,
            'RMSE': np.sqrt(mse_lr),
            'R2': r2_lr,
            'MAE': mae_lr
        }
        print(f"   MSE: {mse_lr:.4f}, RMSE: {np.sqrt(mse_lr):.4f}, R²: {r2_lr:.4f}, MAE: {mae_lr:.4f}")
        
        # Random Forest
        print("\n2. Training Random Forest Regressor...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        self.models['Random Forest'] = rf
        self.results['Random Forest'] = {
            'MSE': mse_rf,
            'RMSE': np.sqrt(mse_rf),
            'R2': r2_rf,
            'MAE': mae_rf
        }
        self.feature_importance['Random Forest'] = rf.feature_importances_
        print(f"   MSE: {mse_rf:.4f}, RMSE: {np.sqrt(mse_rf):.4f}, R²: {r2_rf:.4f}, MAE: {mae_rf:.4f}")
        
        # Gradient Boosting
        print("\n3. Training Gradient Boosting Regressor...")
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        mse_gb = mean_squared_error(y_test, y_pred_gb)
        r2_gb = r2_score(y_test, y_pred_gb)
        mae_gb = mean_absolute_error(y_test, y_pred_gb)
        self.models['Gradient Boosting'] = gb
        self.results['Gradient Boosting'] = {
            'MSE': mse_gb,
            'RMSE': np.sqrt(mse_gb),
            'R2': r2_gb,
            'MAE': mae_gb
        }
        self.feature_importance['Gradient Boosting'] = gb.feature_importances_
        print(f"   MSE: {mse_gb:.4f}, RMSE: {np.sqrt(mse_gb):.4f}, R²: {r2_gb:.4f}, MAE: {mae_gb:.4f}")
    
    def print_summary(self):
        """Print model comparison summary."""
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        
        results_df = pd.DataFrame(self.results).T
        print(results_df.to_string())
        
        best_model = max(self.results, key=lambda x: self.results[x]['R2'])
        print(f"\nBest Model (by R² score): {best_model}")
        print(f"R² Score: {self.results[best_model]['R2']:.4f}")
        
        return best_model
    
    def save_artifacts(self, best_model):
        """Save trained model and preprocessing artifacts."""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the best model
        model_path = os.path.join(models_dir, 'best_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[best_model], f)
        print(f"\nBest model saved to: {model_path}")
        
        # Save the scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save feature names
        # Assuming features from California Housing dataset
        features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        features_path = os.path.join(models_dir, 'features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to: {features_path}")
        
        return model_path, scaler_path, features_path
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance for tree-based models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models_with_importance = ['Random Forest', 'Gradient Boosting']
        
        for idx, model_name in enumerate(models_with_importance):
            importances = self.feature_importance[model_name]
            indices = np.argsort(importances)[::-1]
            
            ax = axes[idx]
            ax.bar(range(len(importances)), importances[indices])
            ax.set_xlabel('Features', fontsize=10)
            ax.set_ylabel('Importance', fontsize=10)
            ax.set_title(f'{model_name} - Feature Importance', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved to: models/feature_importance.png")
        plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("SMART HOUSING VALUATION SYSTEM - MODEL TRAINING")
    print("="*70)
    
    # Initialize pipeline
    pipeline = HousingModelPipeline()
    
    # Load data
    X, y = pipeline.load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = pipeline.preprocess_data(X, y)
    
    # Train models
    pipeline.train_models(X_train, X_test, y_train, y_test)
    
    # Print summary and get best model
    best_model = pipeline.print_summary()
    
    # Save artifacts
    pipeline.save_artifacts(best_model)
    
    # Create visualizations
    pipeline.plot_feature_importance(feature_names)
    
    print("\n" + "="*70)
    print("Training complete! Model artifacts saved in 'models/' directory.")
    print("="*70)


if __name__ == "__main__":
    main()
