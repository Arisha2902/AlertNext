import pandas as pd
import os
# Load your CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load datasets with corrected paths
df = pd.read_csv(os.path.join(current_dir,"PINCODE(India)", "pincode_latlon.csv"))

# Optional: Clean column names if needed
df.columns = [col.strip().lower() for col in df.columns]

# Ask user for pincode
pincode = input("Enter the pincode to search: ")

# Convert to integer for safe comparison
try:
    pincode = int(pincode)
except ValueError:
    print("Invalid pincode!")
    exit()

# Search for the pincode
result = df[df['pincode'] == pincode]

# Show result
if not result.empty:
    lat = result.iloc[0]['latitude']
    lon = result.iloc[0]['longitude']
    print(f"📍 Pincode {pincode} → Latitude: {lat}, Longitude: {lon}")
else:
    print("❌ Pincode not found in the dataset.")