import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def clean_coordinate(coord):
    """
    Clean and validate coordinate values.
    Handles both numeric and string formats (e.g., '20.6N' -> 20.6 or -20.6)
    """
    if isinstance(coord, str):
        direction = coord[-1].upper()
        try:
            value = float(coord[:-1])
            if direction in ['S', 'W']:
                return -value
            return value
        except ValueError:
            logger.warning(f"Invalid coordinate value: {coord}")
            return None
    return coord

def train_cyclone_model(file_path=CYCLONE_DATA_PATH, features=CYCLONE_FEATURES, label='cyclone_risk'):
    """
    Train a cyclone prediction model with proper error handling.
    
    Args:
        file_path (str): Path to the training data
        features (list): List of feature column names
        label (str): Name of the target column
    
    Returns:
        RandomForestClassifier: Trained model
    """
    try:
        # Load and prepare data
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Validate columns
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # Clean coordinates
        for col in ['Latitude', 'Longitude']:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].apply(clean_coordinate)
        
        # Create risk label
        df[label] = df.apply(
            lambda row: 1 if row['Wind Speed (km/h)'] >= 100 or row['Pressure (mb)'] <= 980 else 0,
            axis=1
        )
        
        # Drop missing values
        initial_rows = len(df)
        df.dropna(subset=features + [label], inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")
        
        # Prepare features and target
        X = df[features]
        y = df[label]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            class_weight='balanced', 
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        logger.info(f"\nClassification Report for {label}:")
        logger.info(classification_report(y_test, y_pred, zero_division=0))
        
        return model
        
    except Exception as e:
        logger.error(f"Error in train_cyclone_model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Train the cyclone model
        cyclone_model = train_cyclone_model()
        
        # Save the model
        os.makedirs(os.path.dirname(CYCLONE_MODEL_PATH), exist_ok=True)
        joblib.dump(cyclone_model, CYCLONE_MODEL_PATH)
        logger.info(f"Model saved to {CYCLONE_MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to train and save cyclone model: {str(e)}")
        sys.exit(1)
