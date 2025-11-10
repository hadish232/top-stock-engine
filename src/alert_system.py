import pandas as pd
import numpy as np

def generate_alerts(df, threshold=0.03):
    df['signal'] = np.where(df['predicted_move'] > threshold, "ALERT", "HOLD")
    return df[df['signal']=="ALERT"]
