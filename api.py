from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

# paper=True enables paper trading

load_dotenv()  # reads .env in current working dir
alpaca_secret_key = os.getenv("alpaca_secret_key")
alpaca_key = os.getenv("alpaca_key")

trading_client = TradingClient(alpaca_key, alpaca_secret_key, paper=True)


account = trading_client.get_account()
positions = trading_client.get_all_positions()

print(positions)
