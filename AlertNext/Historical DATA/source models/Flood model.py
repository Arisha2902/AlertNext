import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import logging
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_coordinate(coord):
    """Clean and validate coordinate values."""
    try:
        return float(coord)
    except (ValueError, TypeError):
        logger.warning(f"Invalid coordinate value: {coord}")
        return None

def train_flood_model(file_path=FLOOD_DATA_PATH, features=None, label='flood_risk'):
    """
    Train a flood prediction model with proper error handling.
    
    Args:
        file_path (str): Path to the training data
        features (list): List of feature column names
        label (str): Name of the target column
    
    Returns:
        tuple: (trained_model, scaler)
    """
    try:
        # Load data
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Define features if not provided
        if features is None:
            features = ['Rainfall (mm)', 'Water Level (m)', 'Humidity (%)', 'Latitude', 'Longitude']
        
        # Validate columns
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # Clean coordinates
        for col in ['Latitude', 'Longitude']:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].apply(clean_coordinate)
        
        # Feature engineering
        df['Rainfall_Water_Level'] = df['Rainfall (mm)'] * df['Water Level (m)']
        
        # Add the engineered feature to the features list
        features.append('Rainfall_Water_Level')
        
        # Create the flood_risk column if it doesn't exist
        if label not in df.columns:
            df[label] = df['Flood Occurred'] if 'Flood Occurred' in df.columns else 0
        
        # Drop missing values
        initial_rows = len(df)
        df.dropna(subset=features + [label], inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")
        
        # Features and labels
        X = df[features]
        y = df[label]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        
        # GridSearchCV for RandomForest
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(rf, param_grid, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluation
        y_pred = best_model.predict(X_test)
        logger.info(f"\nClassification Report for {label}:")
        logger.info(classification_report(y_test, y_pred, zero_division=0))
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f}")
        
        return best_model, scaler
        
    except Exception as e:
        logger.error(f"Error in train_flood_model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Train the flood model
        flood_model, scaler = train_flood_model()
        
        # Save the model and scaler
        os.makedirs(os.path.dirname(FLOOD_MODEL_PATH), exist_ok=True)
        joblib.dump(flood_model, FLOOD_MODEL_PATH)
        joblib.dump(scaler, FLOOD_SCALER_PATH)
        logger.info(f"Model saved to {FLOOD_MODEL_PATH}")
        logger.info(f"Scaler saved to {FLOOD_SCALER_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to train and save flood model: {str(e)}")
        sys.exit(1)
