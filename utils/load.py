import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data, name=""):
  try:
    if data is None or data.empty:
      logging.error(f"Data {name} is empty")
    
    filename = f"{name}.csv"
    data.to_csv(filename, index=True)
    
  except Exception as e:
    logging.error(f"Error loading data: {e}")