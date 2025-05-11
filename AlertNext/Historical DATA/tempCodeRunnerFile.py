import os
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = "/ALERTNEXT/PINCODE(India)/pincode_latlon.csv"
# Load datasets with corrected paths
df = pd.read_csv(os.path.join(current_dir,"PINCODE(India)", file_path))