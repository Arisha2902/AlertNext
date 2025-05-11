import pandas as pd
import math
import os
from typing import Optional, List, Dict

class DisasterRiskPredictor:
    def __init__(self):
        self.df = None
        self.risk_df = None
        self.load_data()

    def load_data(self) -> None:
        """Load and validate the required datasets."""
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pincode_path = os.path.join(base_path, "PINCODE(India)", "pincode_latlon.csv")
            risk_path = os.path.join(base_path, "Historical DATA", "data", "combined_disaster_predictions.csv")

            self.df = pd.read_csv(pincode_path)
            self.risk_df = pd.read_csv(
                risk_path,
                dtype={'Latitude': 'float64', 'Longitude': 'float64'},
                low_memory=False
            )

            # Standardize column names
            self.df.columns = [col.strip().lower() for col in self.df.columns]
            self.risk_df.columns = [col.strip().lower() for col in self.risk_df.columns]

            # Validate required columns
            required_pincode_cols = ['pincode', 'latitude', 'longitude']
            required_risk_cols = ['latitude', 'longitude', 'flood_risk', 'wind speed (km/h)', 'magnitude', 'combined_risk']
            
            if not all(col in self.df.columns for col in required_pincode_cols):
                raise ValueError("Missing required columns in pincode dataset")
            if not all(col in self.risk_df.columns for col in required_risk_cols):
                raise ValueError("Missing required columns in risk dataset")

        except FileNotFoundError as e:
            print(f"Error: Required data file not found - {str(e)}")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points."""
        try:
            R = 6371  # Earth's radius in kilometers
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c
        except Exception as e:
            print(f"Error calculating distance: {str(e)}")
            return float('inf')

    def validate_pincode(self, pincode: str) -> bool:
        """Validate Indian pincode format."""
        try:
            pincode_int = int(pincode)
            return len(pincode) == 6 and 100000 <= pincode_int <= 999999
        except ValueError:
            return False

    def check_risk_for_pincode(self) -> None:
        """Check disaster risk for a given pincode."""
        while True:
            pincode = input("📬 Enter the pincode to search (6 digits): ")
            
            if not self.validate_pincode(pincode):
                print("❌ Invalid pincode! Please enter a valid 6-digit Indian pincode.")
                continue

            pincode = int(pincode)
            result = self.df[self.df['pincode'] == pincode]

            if result.empty:
                print("❌ Pincode not found in the dataset.")
                break

            lat = float(result.iloc[0]['latitude'])
            lon = float(result.iloc[0]['longitude'])
            location = f"{result.iloc[0].get('officename', 'Unknown')}, {result.iloc[0].get('statename', 'Unknown')}"

            print(f"\n📍 Location: {location} (Pincode: {pincode})")
            print(f"🌐 Coordinates: Latitude {lat}, Longitude {lon}")

            self._display_nearby_risks(lat, lon)
            break

    def _display_nearby_risks(self, lat: float, lon: float) -> None:
        """Display disaster risks for locations near the given coordinates."""
        tolerance_km = 10
        nearby_risk = []

        for _, row in self.risk_df.iterrows():
            risk_lat = row.get('latitude')
            risk_lon = row.get('longitude')

            if pd.isna(risk_lat) or pd.isna(risk_lon):
                continue

            distance = this.haversine(lat, lon, risk_lat, risk_lon)
            if distance <= tolerance_km:
                nearby_risk.append(row)

        if nearby_risk:
            print("\n✅ Disaster Risk Predictions:")
            for row in nearby_risk:
                self._display_risk_details(row)
        else:
            print("⚠️ No disaster risk prediction found near this location.")

    def _display_risk_details(self, row: pd.Series) -> None:
        """Display formatted risk details for a location."""
        flood_risk = row.get('flood_risk', 'N/A')
        cyclone_risk = row.get('wind speed (km/h)', 'N/A')
        earthquake_risk = row.get('magnitude', 'N/A')
        combined_risk = row.get('combined_risk', 'N/A')

        # Format values
        flood_risk = "N/A" if pd.isna(flood_risk) else round(flood_risk, 2)
        cyclone_risk = "N/A" if pd.isna(cyclone_risk) else round(cyclone_risk)
        earthquake_risk = "N/A" if pd.isna(earthquake_risk) else round(earthquake_risk, 1)
        combined_risk = "N/A" if pd.isna(combined_risk) else int(round(combined_risk))

        print(f"📌 Location (approx): ({round(row['latitude'], 2)}, {round(row['longitude'], 2)})")
        print(f"   🌊 Flood Risk: {flood_risk}")
        print(f"   🌪️ Cyclone Risk: {cyclone_risk}")
        print(f"   🌍 Earthquake Risk: {earthquake_risk}")
        print(f"   📝 Combined Risk (Model): {combined_risk}")
        print("-" * 40)

    def run(self) -> None:
        """Run the main program loop."""
        while True:
            print("\n=== 🔍 Disaster Risk Prediction Menu ===")
            print("1. 🔎 Check disaster risk for a Pincode")
            print("2. 🚪 Exit")
            
            choice = input("Enter your choice (1/2): ")
            
            if choice == '1':
                self.check_risk_for_pincode()
            elif choice == '2':
                print("Goodbye!")
                break
            else:
                print("Invalid choice! Please try again.")

if __name__ == "__main__":
    try:
        predictor = DisasterRiskPredictor()
        predictor.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
