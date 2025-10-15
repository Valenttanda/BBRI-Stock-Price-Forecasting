import yfinance as yf
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_data(start="", stock="", end=None):
  try:
    # Parameter validation
    if not stock:
      logging.error("Input your stock ID")
      raise
    if not start:
      logging.error("Start date must be filled (YYYY-MM-DD)")
    
    stock_data = yf.Ticker(stock)
    if end is None:
      end = datetime.today().strftime('%Y-%m-%d')
    data = stock_data.history(start=start, end=end)
    
    return data
      
  except Exception as e:
    logging.ERROR(f"Scrapping error: {e}")