import pandas as pd
import numpy as np
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

def validate_risk_columns(df, name):
    """Validate and ensure risk columns exist with proper values."""
    risk_columns = {
        'Flood': 'flood_risk',
        'Cyclone': 'cyclone_risk',
        'Earthquake': 'earthquake_risk'
    }
    
    risk_col = risk_columns[name]
    
    # Check if risk column exists
    if risk_col not in df.columns:
        # For Cyclone data
        if name == 'Cyclone':
            logger.info(f"Creating cyclone risk column based on wind speed and pressure")
            df[risk_col] = ((df['Wind Speed (km/h)'] >= 100) | (df['Pressure (mb)'] <= 980)).astype(int)
        
        # For Earthquake data
        elif name == 'Earthquake':
            logger.info(f"Creating earthquake risk column based on magnitude")
            df[risk_col] = (df['Magnitude'] >= 5.0).astype(int)
        
        # For Flood data
        elif name == 'Flood':
            if 'Flood Occurred' in df.columns:
                logger.info(f"Creating flood risk column from 'Flood Occurred'")
                df[risk_col] = df['Flood Occurred'].astype(int)
            else:
                logger.warning(f"No risk column found for {name} data. Creating default risk column.")
                df[risk_col] = 0
    
    # Ensure risk column has proper values (0 or 1)
    df[risk_col] = df[risk_col].fillna(0).astype(int)
    risk_counts = df[risk_col].value_counts()
    logger.info(f"{name} risk distribution - High risk: {risk_counts.get(1, 0)}, Low risk: {risk_counts.get(0, 0)}")
    
    return df

def merge_disaster_predictions():
    """
    Merge predictions from different disaster models into a single dataset.
    Includes proper error handling, data validation, and risk calculation.
    """
    try:
        # Read the individual prediction files
        logger.info("Loading disaster prediction datasets...")
        
        datasets = {
            'Flood': pd.read_csv(FLOOD_DATA_PATH),
            'Cyclone': pd.read_csv(CYCLONE_DATA_PATH),
            'Earthquake': pd.read_csv(EARTHQUAKE_DATA_PATH)
        }
        
        # Log initial data shapes
        for name, df in datasets.items():
            logger.info(f"{name} data shape: {df.shape}")
            
            # Clean coordinates
            for col in ['Latitude', 'Longitude']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        logger.warning(f"{name} data: {null_count} invalid {col} values")
            
            # Validate and ensure risk columns
            datasets[name] = validate_risk_columns(df, name)
        
        # Merge the dataframes on Latitude and Longitude
        logger.info("Merging datasets...")
        merged_df = datasets['Flood']
        for name in ['Cyclone', 'Earthquake']:
            risk_col = f"{name.lower()}_risk"
            merged_df = pd.merge(
                merged_df, 
                datasets[name][['Latitude', 'Longitude', risk_col]], 
                on=['Latitude', 'Longitude'], 
                how='outer'
            )
        
        # Log merge results
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        # Create a combined risk column
        risk_columns = ['flood_risk', 'cyclone_risk', 'earthquake_risk']
        merged_df[risk_columns] = merged_df[risk_columns].fillna(0)
        
        merged_df['combined_risk'] = (
            (merged_df['flood_risk'] == 1) | 
            (merged_df['cyclone_risk'] == 1) | 
            (merged_df['earthquake_risk'] == 1)
        ).astype(int)
        
        # Save the merged predictions
        os.makedirs(os.path.dirname(COMBINED_PREDICTIONS_PATH), exist_ok=True)
        merged_df.to_csv(COMBINED_PREDICTIONS_PATH, index=False)
        logger.info(f"Merged data saved to {COMBINED_PREDICTIONS_PATH}")
        
        # Log final statistics
        for col in risk_columns + ['combined_risk']:
            risk_counts = merged_df[col].value_counts()
            logger.info(f"\n{col} distribution:")
            logger.info(f"High risk locations: {risk_counts.get(1, 0)}")
            logger.info(f"Low risk locations: {risk_counts.get(0, 0)}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error in merge_disaster_predictions: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        merged_df = merge_disaster_predictions()
        logger.info("✅ Successfully merged all disaster predictions")
    except Exception as e:
        logger.error(f"Failed to merge disaster predictions: {str(e)}")
        sys.exit(1)