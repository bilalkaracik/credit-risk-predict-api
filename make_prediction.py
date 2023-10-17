import requests
import pandas as pd

# Folder path
file_path = r"sample_v2.csv"
url = "http://127.0.0.1:5000/predict"

# Convert from CSV to Data Frame
csv_data = pd.read_csv(file_path)

# Convert from Data Frame to Json
json_data = csv_data.to_dict(orient='list')

response = requests.post(url, json=json_data)

print(response.json())
