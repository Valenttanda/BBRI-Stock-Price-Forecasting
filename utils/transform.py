from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transform_data(data):
  try:
    # Convert the 'date' column to datetime format
    cleaned = data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close']]
    cleaned['Date'] = cleaned['Date'].dt.date
    
    return cleaned
  
  except Exception as e:
    logging.error(f"Error transforming data: {e}")