from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import sqlite3
from datetime import datetime, timedelta
import joblib
import pandas as pd
import math
import os
import requests
import json
from geopy.distance import geodesic
import logging
import re
from ngo_data import get_ngos_by_pincode
import random
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('user_registrations.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            pincode TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load models and scalers
try:
    logger.info("Loading models and scalers...")
    flood_model = joblib.load(os.path.join(current_dir, 'models pkl', 'flood_model.pkl'))
    flood_scaler = joblib.load(os.path.join(current_dir, 'models pkl', 'flood_scaler.pkl'))
    earthquake_model = joblib.load(os.path.join(current_dir, 'models pkl', 'earthquake_model.pkl'))
    cyclone_model = joblib.load(os.path.join(current_dir, 'models pkl', 'cyclone_model.pkl'))
    logger.info("Models and scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Load datasets
try:
    logger.info("Loading datasets...")
    df = pd.read_csv(os.path.join(current_dir, 'PINCODE(India)', 'pincode_latlon.csv'))
    flood_data = pd.read_csv(os.path.join(current_dir, 'Historical DATA', 'data', 'Flood.csv'))
    earthquake_data = pd.read_csv(os.path.join(current_dir, 'Historical DATA', 'data', 'Earthquake.csv'))
    cyclone_data = pd.read_csv(os.path.join(current_dir, 'Historical DATA', 'data', 'Cyclone.csv'))
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    flood_data.columns = [col.strip().lower() for col in flood_data.columns]
    earthquake_data.columns = [col.strip().lower() for col in earthquake_data.columns]
    cyclone_data.columns = [col.strip().lower() for col in cyclone_data.columns]
    
    # Ensure required columns exist
    required_columns = {
        'df': ['pincode', 'latitude', 'longitude'],
        'flood_data': ['latitude', 'longitude', 'rainfall (mm)', 'water level (m)', 'humidity (%)'],
        'earthquake_data': ['latitude', 'longitude', 'magnitude'],  # Removed 'depth' as it's not required
        'cyclone_data': ['latitude', 'longitude', 'wind speed (km/h)', 'pressure (mb)']
    }
    
    for dataset_name, columns in required_columns.items():
        dataset = locals()[dataset_name]
        missing_columns = [col for col in columns if col not in dataset.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {dataset_name}: {missing_columns}")
            # Instead of raising an error, we'll handle missing columns gracefully
            for col in missing_columns:
                dataset[col] = 0  # Initialize missing columns with default value
    
    # Convert numeric columns
    numeric_columns = {
        'flood_data': ['rainfall (mm)', 'water level (m)', 'humidity (%)'],
        'earthquake_data': ['magnitude'],  # Removed 'depth'
        'cyclone_data': ['wind speed (km/h)', 'pressure (mb)']
    }
    
    for dataset_name, columns in numeric_columns.items():
        dataset = locals()[dataset_name]
        for col in columns:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce').fillna(0)
    
    # Log data statistics
    logger.info(f"Pincode dataset loaded with {len(df)} entries")
    logger.info(f"Flood dataset loaded with {len(flood_data)} entries")
    logger.info(f"Earthquake dataset loaded with {len(earthquake_data)} entries")
    logger.info(f"Cyclone dataset loaded with {len(cyclone_data)} entries")
    
    # Log sample data
    logger.debug("Sample flood data:")
    logger.debug(flood_data.head())
    logger.debug("Sample earthquake data:")
    logger.debug(earthquake_data.head())
    logger.debug("Sample cyclone data:")
    logger.debug(cyclone_data.head())
except Exception as e:
    logger.error(f"Error loading datasets: {str(e)}")
    raise

# Load pincode data
risk_df = pd.read_csv(
    os.path.join(current_dir, 'Historical DATA', 'data', 'combined_disaster_predictions.csv'),
    dtype={'Latitude': 'float64', 'Longitude': 'float64'},
    low_memory=False
)

# Log data statistics
logger.debug(f"Risk dataset loaded with {len(risk_df)} entries")
logger.debug("Sample risk data:")
logger.debug(risk_df.head())

# Pydantic models for request body validation
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
class PincodeRequest(BaseModel):
    pincode: int     

class PincodeRequest1(BaseModel):
    pincode: str = Field(..., min_length=6, max_length=6)
    @validator('pincode')
    def validate_pincode(cls, value):
        if not re.match(r'^\d{6}$', value):
            raise ValueError('PIN code must be a 6-digit number')
        return value

# Clean lat/lon if strings (for your data pre-processing)
def clean_coordinate(coord):
    if isinstance(coord, str):
        direction = coord[-1].upper()
        try:
            value = float(coord[:-1])
            if direction in ['S', 'W']:
                return -value
            return value
        except ValueError:
            return None
    return coord

# Distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)  # Fixed: was using lat2 instead of lon2
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_seasonal_factors(latitude):
    # Simple seasonal factors based on latitude
    if latitude > 23.5:  # Northern hemisphere
        monsoon_factor = 1.5 if 6 <= datetime.now().month <= 9 else 0.8
        cyclone_factor = 1.3 if 4 <= datetime.now().month <= 6 else 0.7
    else:  # Southern hemisphere
        monsoon_factor = 1.5 if 12 <= datetime.now().month <= 3 else 0.8
        cyclone_factor = 1.3 if 10 <= datetime.now().month <= 12 else 0.7
    return monsoon_factor, cyclone_factor

def get_location_data(latitude, longitude, data, radius=0.1):
    """Get data points within a radius of the given location"""
    mask = (
        (data['Latitude'] >= latitude - radius) & 
        (data['Latitude'] <= latitude + radius) & 
        (data['Longitude'] >= longitude - radius) & 
        (data['Longitude'] <= longitude + radius)
    )
    return data[mask]

async def predict_flood_risk(pincode: str):
    try:
        # Convert pincode to string and ensure it's 6 digits
        pincode = str(pincode).zfill(6)
        
        # Get location data
        location_data = df[df['pincode'] == pincode]
        if location_data.empty:
            return {"error": "Pincode not found", "risk": 0}
        
        lat, lon = float(location_data.iloc[0]['latitude']), float(location_data.iloc[0]['longitude'])
        
        # Find nearest flood data point with distance calculation
        flood_data['distance'] = flood_data.apply(
            lambda row: haversine(lat, lon, float(row['latitude']), float(row['longitude'])), axis=1
        )
        nearest_flood = flood_data.loc[flood_data['distance'].idxmin()]
        
        # Log the distance for debugging
        logger.debug(f"Nearest flood point distance: {nearest_flood['distance']} km")
        
        return {
            "risk": float(nearest_flood['rainfall (mm)']),
            "water_level": float(nearest_flood['water level (m)']),
            "humidity": float(nearest_flood['humidity (%)']),
            "distance": float(nearest_flood['distance'])
        }
    except Exception as e:
        logger.error(f"Error in flood prediction: {str(e)}")
        return {"error": str(e), "risk": 0}

async def predict_earthquake_risk(pincode: str):
    try:
        # Convert pincode to string and ensure it's 6 digits
        pincode = str(pincode).zfill(6)
        
        # Get location data
        location_data = df[df['pincode'] == pincode]
        if location_data.empty:
            return {"error": "Pincode not found", "risk": 0}
        
        lat, lon = float(location_data.iloc[0]['latitude']), float(location_data.iloc[0]['longitude'])
        
        # Find nearest earthquake data point with distance calculation
        earthquake_data['distance'] = earthquake_data.apply(
            lambda row: haversine(lat, lon, float(row['latitude']), float(row['longitude'])), axis=1
        )
        nearest_eq = earthquake_data.loc[earthquake_data['distance'].idxmin()]
        
        # Log the distance for debugging
        logger.debug(f"Nearest earthquake point distance: {nearest_eq['distance']} km")
        
        return {"risk": float(nearest_eq['magnitude']), "distance": float(nearest_eq['distance'])}
    except Exception as e:
        logger.error(f"Error in earthquake prediction: {str(e)}")
        return {"error": str(e), "risk": 0}

async def predict_cyclone_risk(pincode: str):
    try:
        # Convert pincode to string and ensure it's 6 digits
        pincode = str(pincode).zfill(6)
        
        # Get location data
        location_data = df[df['pincode'] == pincode]
        if location_data.empty:
            return {"error": "Pincode not found", "risk": 0}
        
        lat, lon = float(location_data.iloc[0]['latitude']), float(location_data.iloc[0]['longitude'])
        
        # Find nearest cyclone data point with distance calculation
        cyclone_data['distance'] = cyclone_data.apply(
            lambda row: haversine(lat, lon, float(row['latitude']), float(row['longitude'])), axis=1
        )
        nearest_cyclone = cyclone_data.loc[cyclone_data['distance'].idxmin()]
        
        # Log the distance for debugging
        logger.debug(f"Nearest cyclone point distance: {nearest_cyclone['distance']} km")
        
        return {
            "wind_speed": float(nearest_cyclone['wind speed (km/h)']),
            "pressure": float(nearest_cyclone['pressure (mb)']),
            "distance": float(nearest_cyclone['distance'])
        }
    except Exception as e:
        logger.error(f"Error in cyclone prediction: {str(e)}")
        return {"error": str(e), "risk": 0}

async def get_risk_by_pincode(pincode: str):
    try:
        # Convert pincode to string and ensure it's 6 digits
        pincode = str(pincode).zfill(6)
        logger.debug(f"Searching for pincode: {pincode}")
        
        # Convert pincode column to string for comparison
        df['pincode'] = df['pincode'].astype(str).str.zfill(6)
        
        # Find the pincode in the dataset
        location_data = df[df['pincode'] == pincode]
        
        if location_data.empty:
            logger.warning(f"Pincode {pincode} not found in dataset")
            return {
                "error": "Pincode not found",
                "location": None,
                "coordinates": None,
                "predictions": {
                    "flood": {"rainfall": 0, "water_level": 0, "humidity": 0, "distance": 0},
                    "earthquake": {"magnitude": 0, "distance": 0},
                    "cyclone": {"wind_speed": 0, "pressure": 0, "distance": 0}
                }
            }
        
        lat = float(location_data.iloc[0]['latitude'])
        lon = float(location_data.iloc[0]['longitude'])
        
        # Get predictions
        flood_risk = await predict_flood_risk(pincode)
        earthquake_risk = await predict_earthquake_risk(pincode)
        cyclone_risk = await predict_cyclone_risk(pincode)
        
        return {
            "location": {
                "pincode": pincode,
                "latitude": lat,
                "longitude": lon
            },
            "coordinates": {
                "latitude": lat,
                "longitude": lon
            },
            "predictions": {
                "flood": {
                    "rainfall": flood_risk.get("risk", 0),
                    "water_level": flood_risk.get("water_level", 0),
                    "humidity": flood_risk.get("humidity", 0),
                    "distance": flood_risk.get("distance", 0)
                },
                "earthquake": {
                    "magnitude": earthquake_risk.get("risk", 0),
                    "distance": earthquake_risk.get("distance", 0)
                },
                "cyclone": {
                    "wind_speed": cyclone_risk.get("wind_speed", 0),
                    "pressure": cyclone_risk.get("pressure", 0),
                    "distance": cyclone_risk.get("distance", 0)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in get_risk_by_pincode: {str(e)}")
        return {
            "error": str(e),
            "location": None,
            "coordinates": None,
            "predictions": {
                "flood": {"rainfall": 0, "water_level": 0, "humidity": 0, "distance": 0},
                "earthquake": {"magnitude": 0, "distance": 0},
                "cyclone": {"wind_speed": 0, "pressure": 0, "distance": 0}
            }
        }

# Handle favicon.ico requests
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Predict disaster risk for a specific location
@app.post("/predict_flood_risk")
async def predict_flood(data: PredictionRequest):
    try:
        # Find nearest pincode to the given coordinates
        df['distance'] = df.apply(
            lambda row: haversine(data.latitude, data.longitude, float(row['latitude']), float(row['longitude'])),
            axis=1
        )
        nearest_pincode = df.loc[df['distance'].idxmin(), 'pincode']
        risk = await predict_flood_risk(nearest_pincode)
        return risk
    except Exception as e:
        logger.error(f"Error in flood prediction: {str(e)}")
        return {"risk": None}

@app.post("/predict_earthquake_risk")
async def predict_earthquake(data: PredictionRequest):
    try:
        # Find nearest pincode to the given coordinates
        df['distance'] = df.apply(
            lambda row: haversine(data.latitude, data.longitude, float(row['latitude']), float(row['longitude'])),
            axis=1
        )
        nearest_pincode = df.loc[df['distance'].idxmin(), 'pincode']
        risk = await predict_earthquake_risk(nearest_pincode)
        return risk
    except Exception as e:
        logger.error(f"Error in earthquake prediction: {str(e)}")
        return {"risk": None}

@app.post("/predict_cyclone_risk")
async def predict_cyclone(data: PredictionRequest):
    try:
        # Find nearest pincode to the given coordinates
        df['distance'] = df.apply(
            lambda row: haversine(data.latitude, data.longitude, float(row['latitude']), float(row['longitude'])),
            axis=1
        )
        nearest_pincode = df.loc[df['distance'].idxmin(), 'pincode']
        risk = await predict_cyclone_risk(nearest_pincode)
        return risk
    except Exception as e:
        logger.error(f"Error in cyclone prediction: {str(e)}")
        return {"risk": None}

@app.post("/predict_by_pincode")
async def predict_by_pincode(data: PincodeRequest):
    try:
        result = await get_risk_by_pincode(data.pincode)
        return result
    except Exception as e:
        logger.error(f"Error in predict_by_pincode: {str(e)}")
        return {
            "error": str(e),
            "location": "Unknown",
            "coordinates": {"latitude": 0, "longitude": 0},
            "predictions": {
                "flood": {"rainfall": 0, "water_level": 0, "humidity": 0, "distance": 0},
                "earthquake": {"magnitude": 0, "distance": 0},
                "cyclone": {"wind_speed": 0, "pressure": 0, "distance": 0}
            }
        }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint to get NGOs by PIN code
@app.post("/api/ngos/")
async def get_ngos(pincode_request: PincodeRequest1):
    try:
        pincode = pincode_request.pincode  # Updated field name
        logger.debug(f"Searching for NGOs near PIN code: {pincode}")
        
        # Get NGOs for the given PIN code
        ngos = get_ngos_by_pincode(pincode)
        
        if not ngos:
            return JSONResponse(
                status_code=404,
                content={"message": f"No NGOs found near PIN code {pincode}"}
            )
        
        return {"ngos": ngos}
    
    except ValueError as e:
        logger.error(f"Invalid PIN code format: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"message": str(e)}
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "An unexpected error occurred. Please try again later."}
        )

@app.get("/predict")
async def predict(latitude: float, longitude: float):
    try:
        flood_risk = await predict_flood_risk(latitude)
        earthquake_risk = await predict_earthquake_risk(latitude)
        cyclone_risk = await predict_cyclone_risk(latitude)
        
        return {
            "flood_risk": flood_risk.get("risk", 0),
            "earthquake_risk": earthquake_risk.get("risk", 0),
            "cyclone_risk": cyclone_risk.get("risk", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
async def register_user(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    pincode: str = Form(...)
):
    try:
        conn = sqlite3.connect('user_registrations.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO users (name, email, phone, pincode)
            VALUES (?, ?, ?, ?)
        ''', (name, email, phone, pincode))
        conn.commit()
        conn.close()
        return {"message": "Registration successful"}
    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
