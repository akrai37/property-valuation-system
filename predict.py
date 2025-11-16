"""
Smart Housing Valuation System - Prediction Utility
Provides a simple interface for making single predictions.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


class PricePredictorUtil:
    """Utility class for loading model and making predictions."""
    
    def __init__(self, models_dir='models'):
        """Initialize the predictor with saved artifacts."""
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.features = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained model, scaler, and feature names."""
        try:
            with open(self.models_dir / "best_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            
            with open(self.models_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            with open(self.models_dir / "features.pkl", "rb") as f:
                self.features = pickle.load(f)
            
            print("✓ Model artifacts loaded successfully!")
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print("Please run 'python train_model.py' first to generate model artifacts.")
            raise
    
    def predict(self, **kwargs):
        """
        Make a price prediction given property features.
        
        Parameters:
        -----------
        MedInc : float
            Median income in units of $10,000
        HouseAge : int
            Median house age in years
        AveRooms : float
            Average number of rooms per household
        AveBedrms : float
            Average number of bedrooms per household
        Population : int
            Block group population
        AveOccup : float
            Average household occupancy
        Latitude : float
            Block group latitude
        Longitude : float
            Block group longitude
        
        Returns:
        --------
        dict : Dictionary with prediction and input values
        """
        # Extract values in the correct order
        feature_values = []
        for feature in self.features:
            if feature not in kwargs:
                raise ValueError(f"Missing required feature: {feature}")
            feature_values.append(kwargs[feature])
        
        # Convert to array and scale
        input_data = np.array([feature_values])
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        price_in_dollars = prediction * 100000
        
        return {
            'predicted_price': price_in_dollars,
            'prediction_value': prediction,
            'input_features': dict(zip(self.features, feature_values))
        }
    
    def predict_batch(self, dataframe):
        """
        Make predictions for a batch of properties.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            DataFrame with columns matching feature names
        
        Returns:
        --------
        pd.Series : Predictions for each row
        """
        feature_data = dataframe[self.features].values
        scaled_data = self.scaler.transform(feature_data)
        predictions = self.model.predict(scaled_data) * 100000
        return pd.Series(predictions, name='predicted_price')


def example_predictions():
    """Run example predictions."""
    print("\n" + "="*70)
    print("SMART HOUSING VALUATION - EXAMPLE PREDICTIONS")
    print("="*70 + "\n")
    
    try:
        predictor = PricePredictorUtil()
        
        # Example properties
        examples = [
            {
                'name': 'Luxury Home - High Income Area',
                'features': {
                    'MedInc': 8.0,
                    'HouseAge': 20,
                    'AveRooms': 7.0,
                    'AveBedrms': 3.5,
                    'Population': 5000,
                    'AveOccup': 3.0,
                    'Latitude': 37.5,
                    'Longitude': -120.5
                }
            },
            {
                'name': 'Budget Home - Urban Area',
                'features': {
                    'MedInc': 2.0,
                    'HouseAge': 40,
                    'AveRooms': 4.0,
                    'AveBedrms': 1.5,
                    'Population': 12000,
                    'AveOccup': 4.0,
                    'Latitude': 37.0,
                    'Longitude': -121.0
                }
            },
            {
                'name': 'Modern Home - Suburban',
                'features': {
                    'MedInc': 5.0,
                    'HouseAge': 10,
                    'AveRooms': 6.0,
                    'AveBedrms': 3.0,
                    'Population': 8000,
                    'AveOccup': 3.5,
                    'Latitude': 37.8,
                    'Longitude': -119.5
                }
            }
        ]
        
        for example in examples:
            print(f"\n📍 {example['name']}")
            print("-" * 70)
            
            result = predictor.predict(**example['features'])
            
            print(f"\nFeatures:")
            for feature, value in result['input_features'].items():
                print(f"  • {feature:12s}: {value:>10.2f}")
            
            print(f"\n💰 Predicted Price: ${result['predicted_price']:>12,.2f}")
            print(f"   (Raw value: {result['prediction_value']:.4f})")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_predictions()
