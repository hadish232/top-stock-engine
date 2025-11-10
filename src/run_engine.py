# run_engine.py
from src.engine import engine_run
import os
from datetime import datetime
os.makedirs("outputs", exist_ok=True)
# replace with your universe (example)
UNIVERSE = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "KOTAKBANK", "AXISBANK"]
out = engine_run(UNIVERSE)
today = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
out_path = f"outputs/topstocks_{today}.csv"
out.to_csv(out_path, index=False)
print("Saved", out_path)
