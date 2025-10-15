from extract import scrape_data
from transform import transform_data
from load import load_data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
  try:
    # Load data from .env
    stock = "BBRI.JK"
    name = "BBRI_Stock"
    start = "2020-01-01"
    
    # Extracting data
    logging.info("Extracting data...")
    data = scrape_data(start=start, stock=stock)
    logging.info("Data extracted successfully.")
    
    # Transforming data
    logging.info("Transforming data...")
    transformed_data = transform_data(data)
    logging.info("Data transformed successfully.")
    
    # Loading data
    logging.info("Loading data...")
    load_data(transformed_data, name=name)
    logging.info("Data loaded successfully.")
  
  except Exception as e:
    logging.error(f"An error occurred: {e}")  
    
if __name__ == "__main__":
  main()