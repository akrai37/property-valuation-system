"""
Smart Housing Valuation System - Web Application
Interactive Streamlit app for housing price predictions.
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Smart Housing Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional production-grade UI with dark theme support
st.markdown("""
    <style>
    /* Root variables and dark mode support */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-color: #667eea;
    }
    
    /* Light theme */
    [data-theme="light"] {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #f0f2f6;
        --text-primary: #1a1a1a;
        --text-secondary: #4a5568;
        --text-tertiary: #718096;
        --border-color: #e2e8f0;
    }
    
    /* Dark theme */
    [data-theme="dark"] {
        --bg-primary: #1e1e1e;
        --bg-secondary: #2d2d2d;
        --bg-tertiary: #3a3a3a;
        --text-primary: #ffffff;
        --text-secondary: #e0e0e0;
        --text-tertiary: #b0b0b0;
        --border-color: #404040;
    }
    
    /* Main container - adaptive background */
    .main {
        padding: 2rem;
    }
    
    /* Light theme main background */
    [data-theme="light"] .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Dark theme main background */
    [data-theme="dark"] .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem 0;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #667eea;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    [data-theme="dark"] .header-subtitle {
        color: #99b0ff;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    [data-theme="light"] h1, [data-theme="light"] h2, [data-theme="light"] h3 {
        color: #1a1a1a;
    }
    
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3 {
        color: #ffffff;
    }
    
    p, span, label {
        [data-theme="light"] & {
            color: #4a5568;
        }
        [data-theme="dark"] & {
            color: #e0e0e0;
        }
    }
    
    /* Metric cards - professional styling */
    .stMetric {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--border-color);
        border-left: 4px solid #667eea;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stMetric:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
        border-left-color: #764ba2;
    }
    
    .stMetricLabel {
        color: var(--text-tertiary);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetricValue {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 800;
    }
    
    /* Prediction result box - premium styling */
    .prediction-box {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(76, 175, 80, 0.25);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    }
    
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin: 0.5rem 0;
    }
    
    .prediction-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input sliders */
    .stSlider {
        padding: 1.5rem 0;
    }
    
    /* Buttons - premium styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 2.5rem;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: rgba(102, 126, 234, 0.1);
    }
    
    /* Cards and containers */
    .card {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* Data tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Feature info boxes */
    .feature-box {
        background: var(--bg-tertiary);
        padding: 1.25rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border-left: 4px solid #667eea;
        border: 1px solid var(--border-color);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .feature-name {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--bg-secondary);
    }
    
    /* Text colors for dark theme */
    [data-theme="dark"] {
        color: #ffffff;
    }
    
    /* Links */
    a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.2s ease;
    }
    
    a:hover {
        color: #764ba2;
    }
    
    [data-theme="dark"] a {
        color: #99b0ff;
    }
    
    [data-theme="dark"] a:hover {
        color: #c5d5ff;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    [data-theme="dark"] .info-box {
        background: rgba(102, 126, 234, 0.15);
    }
    
    /* Success boxes */
    .success-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and feature names."""
    models_dir = Path(__file__).parent / "models"
    
    try:
        with open(models_dir / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(models_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(models_dir / "features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, scaler, features
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run `train_model.py` first.")
        st.stop()


def get_feature_importance():
    """Load feature importance for visualization."""
    models_dir = Path(__file__).parent / "models"
    try:
        from PIL import Image
        img = Image.open(models_dir / "feature_importance.png")
        return img
    except FileNotFoundError:
        return None


def main():
    """Main application."""
    # Professional Production Header
    st.markdown("""
        <div class="header-container">
            <div class="header-title">🏠 Smart Housing Valuation</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #4caf50; margin-top: 0.5rem; margin-bottom: 1.2rem;">
                💳 Real Estate Price Prediction
            </div>
            <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%); 
                        padding: 1.2rem; border-radius: 10px; border-left: 4px solid #4caf50; margin-top: 1rem;">
                <p style="margin: 0; font-size: 1rem; line-height: 1.6; color: #333; font-weight: 500;">
                    Get instant property valuations in seconds. Simply adjust property features to receive accurate price predictions based on market data.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, features = load_model_artifacts()
    
    # Initialize session state for reset
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    
    # Create tabs with enhanced styling
    tab1, tab2, tab3 = st.tabs([
        "🏠 **PROPERTY PREDICTION**", 
        "📊 **FEATURE IMPORTANCE**", 
        "ℹ️ **PROJECT INFO**"
    ])
    
    with tab1:
        st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid var(--border-color);">
                <h2 style="margin-top: 0;">💰 Property Valuation</h2>
                <p style="margin: 0; opacity: 0.9;">Adjust property features to get an instant AI-powered price prediction</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="feature-box">
                    <h4 style="margin-top: 0; color: #667eea;">🏘️ Property Features</h4>
                </div>
            """, unsafe_allow_html=True)
            
            medinc = st.slider(
                "💵 Median Income",
                min_value=0.5,
                max_value=15.0,
                value=3.0,
                step=0.1,
                help="Median household income: 3 = $30k, 5 = $50k, 10 = $100k",
                key="income"
            )
            
            house_age = st.slider(
                "📅 House Age",
                min_value=1,
                max_value=52,
                value=25,
                step=1,
                help="Age of the house in years",
                key="age"
            )
            
            ave_rooms = st.slider(
                "🚪 Average Rooms",
                min_value=1.0,
                max_value=10.0,
                value=5.5,
                step=0.1,
                help="Average number of rooms per household",
                key="rooms"
            )
            
            ave_bedrms = st.slider(
                "🛏️ Average Bedrooms",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Average number of bedrooms per household",
                key="beds"
            )
        
        with col2:
            st.markdown("""
                <div class="feature-box">
                    <h4 style="margin-top: 0; color: #667eea;">📍 Location & Demographics</h4>
                </div>
            """, unsafe_allow_html=True)
            
            population = st.slider(
                "👥 Population",
                min_value=100,
                max_value=35000,
                value=1500,
                step=100,
                help="Number of residents in the neighborhood block",
                key="pop"
            )
            
            ave_occup = st.slider(
                "👨‍👩‍👧‍👦 Average Occupancy",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Average number of people per household",
                key="occup"
            )
            
            latitude = st.slider(
                "🧭 Latitude",
                min_value=32.0,
                max_value=42.0,
                value=37.5,
                step=0.1,
                help="Geographic latitude (North-South position)",
                key="lat"
            )
            
            longitude = st.slider(
                "🧭 Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-120.0,
                step=0.1,
                help="Geographic longitude (East-West position)",
                key="lon"
            )
        
        # Divider and prediction button
        st.divider()
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([0.8, 1, 0.8])
        with col_btn2:
            if st.button("🚀 Generate Valuation", use_container_width=True, key="predict_btn"):
                st.session_state.show_prediction = True
                input_data = np.array([[
                    medinc, house_age, ave_rooms, ave_bedrms,
                    population, ave_occup, latitude, longitude
                ]])
                
                # Scale input
                input_scaled = scaler.transform(input_data)
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                price_in_dollars = prediction * 100000  # Convert to actual price
                
                # Store prediction data in session state
                st.session_state.prediction_data = {
                    'price': price_in_dollars,
                    'medinc': medinc,
                    'house_age': house_age,
                    'ave_rooms': ave_rooms,
                    'population': population,
                    'input_data': input_data,
                    'features': features
                }
        
        # Display prediction if available
        if st.session_state.show_prediction and st.session_state.prediction_data:
            data = st.session_state.prediction_data
            
            # Display prediction with professional styling
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">💎 Estimated Property Value</div>
                <div class="prediction-amount">${data['price']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display input summary
            st.divider()
            st.subheader("📋 Property Summary")
            
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            with col_summary1:
                st.metric("💵 Income", f"${data['medinc'] * 10000:,.0f}")
            with col_summary2:
                st.metric("📅 Age", f"{data['house_age']} yrs")
            with col_summary3:
                st.metric("🚪 Rooms", f"{data['ave_rooms']:.1f}")
            with col_summary4:
                st.metric("👥 Population", f"{int(data['population']):,}")
            
            st.divider()
            st.subheader("📊 Complete Feature Breakdown")
            
            # Create mapping of short names to full names
            feature_names_mapping = {
                'MedInc': '💰 Median Income (in $10,000 units)',
                'HouseAge': '📅 House Age (years)',
                'AveRooms': '🚪 Average Rooms per Household',
                'AveBedrms': '🛏️ Average Bedrooms per Household',
                'Population': '👥 Total Population',
                'AveOccup': '👨‍👩‍👧‍👦 Average Occupancy (people per household)',
                'Latitude': '🧭 Latitude (North-South Position)',
                'Longitude': '🧭 Longitude (East-West Position)'
            }
            
            full_feature_names = [feature_names_mapping.get(f, f) for f in data['features']]
            
            input_df = pd.DataFrame({
                'Feature': full_feature_names,
                'Value': data['input_data'][0]
            })
            
            st.dataframe(input_df, use_container_width=True, hide_index=True)
            
            # Reset button
            st.divider()
            col_reset1, col_reset2, col_reset3 = st.columns([1.5, 1, 1.5])
            with col_reset2:
                if st.button("🔄 Reset & Clear", use_container_width=True, key="reset_btn"):
                    st.session_state.show_prediction = False
                    st.session_state.prediction_data = None
                    st.rerun()
    
    with tab2:
        st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid var(--border-color);">
                <h2 style="margin-top: 0;">📈 Feature Importance Analysis</h2>
                <p style="margin: 0; opacity: 0.9;">Understand which factors drive property valuation in our AI model</p>
            </div>
        """, unsafe_allow_html=True)
        
        img = get_feature_importance()
        if img:
            col_img1, col_img2 = st.columns([2, 1.2])
            with col_img1:
                st.image(img, use_container_width=True)
            with col_img2:
                st.markdown("""
                    <div class="feature-box">
                        <h4 style="margin-top: 0;">📊 What This Shows</h4>
                        <p style="font-size: 0.9rem; margin: 0;">
                            Feature importance scores show how much each input variable contributes to predictions. 
                            Longer bars = greater impact on final valuations.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            st.subheader("🔍 Feature Impact Guide")
            
            st.markdown("""
            **Why Only 2 Models in the Chart Above?**
            
            The chart shows **Random Forest and Gradient Boosting** because they are **tree-based models** 
            that calculate feature importance differently than Linear Regression:
            
            - 🌲 **Tree-Based Models** (Random Forest & Gradient Boosting): 
              Have built-in `feature_importances_` that measure how often each feature splits the decision trees
            
            - 📊 **Linear Regression**: 
              Uses coefficient weights, not comparable "importance" scores - it's a different metric entirely
            
            This is why only the 2 tree-based models are visualized together for fair comparison.
            """)
            
            st.markdown("---")
            st.subheader("📌 What Each Feature Means")
            
            features_guide = {
                "💰 Median Income": "Strongest predictor - directly correlates with property values",
                "🧭 Latitude": "Geographic position (North-South) - location is critical",
                "🧭 Longitude": "Geographic position (East-West) - complements latitude for location value",
                "📅 House Age": "Construction year affects desirability and market value",
                "👥 Population": "Area density and housing demand indicators",
                "🚪 Average Rooms": "Property size - more rooms generally mean higher prices",
                "🛏️ Average Bedrooms": "Bedroom count is a key market valuation factor",
                "👨‍👩‍👧‍👦 Average Occupancy": "Household composition suggests neighborhood characteristics"
            }
            
            for feature, description in features_guide.items():
                st.markdown(f"**{feature}**: {description}")
        else:
            st.warning("⚠️ Feature importance plot not found. Run `python train_model.py` first.")
    
    with tab3:
        st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid var(--border-color);">
                <h2 style="margin-top: 0;">ℹ️ Project Information</h2>
                <p style="margin: 0; opacity: 0.9;">Understand the technology, data, and methodology behind this valuation system</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🎯 Project Overview")
        st.markdown("""
        The **Smart Housing Valuation System** uses state-of-the-art machine learning to predict 
        residential property values. Our Random Forest model achieves **80.5% accuracy** on test data.
        
        **Use Cases:**
        - 🏠 Real estate professionals making market assessments
        - 📊 Investors evaluating property portfolios
        - 🎯 Homebuyers understanding fair market prices
        - 📈 Data scientists learning ML applications
        """)
        
        st.divider()
        
        st.subheader("🤖 Machine Learning Models Comparison")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown("""
            <div class="feature-box">
                <h4 style="margin-top: 0;">📈 Linear Regression</h4>
                <p style="font-size: 0.85rem;">
                    <strong>R² Score:</strong> 57.6%<br>
                    Fast baseline model. Best for understanding simple relationships.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown("""
            <div class="feature-box">
                <h4 style="margin-top: 0;">🌲 Random Forest ⭐</h4>
                <p style="font-size: 0.85rem;">
                    <strong>R² Score:</strong> 80.5%<br>
                    <span style="color: #667eea; font-weight: 600;">Selected Model</span><br>
                    Captures non-linear patterns effectively.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown("""
            <div class="feature-box">
                <h4 style="margin-top: 0;">⚡ Gradient Boosting</h4>
                <p style="font-size: 0.85rem;">
                    <strong>R² Score:</strong> 77.6%<br>
                    Powerful ensemble method for complex patterns.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("📊 Dataset Information")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("📁 Total Records", "20,640")
        with col_d2:
            st.metric("🎯 Features", "8")
        with col_d3:
            st.metric("📍 Geographic Region", "California")
        with col_d4:
            st.metric("📅 Census Year", "1990")
        
        st.divider()
        
        st.subheader("🔧 Input Features Guide")
        
        col_property, col_location = st.columns(2)
        
        with col_property:
            st.markdown("""
            **🏠 Property Features:**
            
            • **Median Income** ($5k - $150k)
              - Household income level
              - Strongest valuation driver
            
            • **House Age** (1-52 years)
              - Years since construction
              - Affects desirability
            
            • **Average Rooms** (1-10)
              - Rooms per household
              - Property size indicator
            
            • **Average Bedrooms** (0.5-5)
              - Bedroom count
              - Market preference factor
            """)
        
        with col_location:
            st.markdown("""
            **📍 Location & Demographics:**
            
            • **Latitude** (32°-42°N)
              - North-South position
              - Critical location factor
            
            • **Longitude** (114°-125°W)
              - East-West position
              - Completes location data
            
            • **Population** (100-35,000)
              - Area population
              - Demand indicator
            
            • **Avg Occupancy** (1-10)
              - People per household
              - Neighborhood characteristic
            """)
    
    with tab3:
        st.header("ℹ️ About This Project")
        
        st.subheader("🎯 Project Overview")
        st.markdown("""
        The **Smart Housing Valuation System** uses advanced machine learning to predict 
        residential property prices based on key housing and location features.
        
        Perfect for:
        - 🏠 Real estate professionals
        - 📊 Market analysis
        - 🎯 Investment decisions
        - 📈 Property valuations
        """)
        
        st.subheader("📚 ML Models Used")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div class="feature-box">
                <h4>📈 Linear Regression</h4>
                <p style="font-size: 0.9rem;">Fast baseline model for understanding linear relationships between features and price.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div class="feature-box">
                <h4>🌲 Random Forest</h4>
                <p style="font-size: 0.9rem;">Ensemble method that captures non-linear patterns and interactions.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div class="feature-box">
                <h4>⚡ Gradient Boosting</h4>
                <p style="font-size: 0.9rem;">Advanced ensemble for highest-accuracy predictions.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📊 Dataset Information")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("📁 Records", "20,640")
        with col_d2:
            st.metric("🎯 Features", "8")
        with col_d3:
            st.metric("📍 Region", "California")
        with col_d4:
            st.metric("📅 Year", "1990 Census")
        
        st.markdown("---")
        
        st.subheader("🔧 Input Features Explained")
        
        col_feature1, col_feature2 = st.columns(2)
        
        with col_feature1:
            st.markdown("""
            **Property Features:**
            
            💵 **Median Income**: Household income in $10,000 units
            - Range: $5,000 - $150,000
            - Strongest price predictor
            
            📅 **House Age**: Years since construction
            - Range: 1-52 years
            - Newer homes have different patterns
            
            🚪 **Average Rooms**: Rooms per household
            - Range: 1-10 rooms
            - Correlates with property size
            
            🛏️ **Average Bedrooms**: Bedrooms per household
            - Range: 0.5-5 bedrooms
            - Market relevance varies by location
            """)
        
        with col_feature2:
            st.markdown("""
            **Location & Demographics:**
            
            👥 **Population**: Block group residents
            - Range: 100-35,000 people
            - Density affects valuations
            
            👨‍👩‍👧‍👦 **Average Occupancy**: People per household
            - Range: 1-10 people
            - Indicates area development
            
            🧭 **Latitude & Longitude**: Geographic coordinates
            - California (32°-42° N, 114°-125° W)
            - Location is crucial for pricing
            """)
        
        st.markdown("---")
        
        st.subheader("📝 Quick Start Guide")
        with st.expander("👉 Click to expand step-by-step instructions"):
            st.markdown("""
            1. **Go to Prediction Tab**: Click the "📊 Prediction" tab above
            2. **Adjust Features**: Use the sliders to set property characteristics
            3. **Predict**: Click the "🎯 Predict Price" button
            4. **View Results**: See the estimated price and property summary
            5. **Explore**: Check the "📈 Feature Importance" tab to understand factors
            """)
        
        st.subheader("💡 Pro Tips for Best Results")
        col_tip1, col_tip2 = st.columns(2)
        with col_tip1:
            st.markdown("""
            ✅ **Do:**
            - Use realistic feature values
            - Consider location heavily
            - Factor in income levels
            - Think about house condition (age)
            """)
        with col_tip2:
            st.markdown("""
            ⚠️ **Remember:**
            - Model trained on 1990 data
            - Not all market factors included
            - Use as reference tool
            - Verify with market experts
            """)


if __name__ == "__main__":
    main()
