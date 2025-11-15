
"""generate_data.py
Generate a synthetic time-series dataset for power/load monitoring.
Produces CSV at ../data/synthetic_timeseries.csv
Usage:
    python generate_data.py --n 12000 --freq_seconds 60 --anomaly_fraction 0.03
"""
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import json

np.random.seed(42)

def base_load(t_idx):
    day_frac = (t_idx % (24*60)) / (24*60)
    daily = 100 + 40*np.sin(2*np.pi*day_frac - 0.5)
    weekly = 5*np.sin(2*np.pi*(t_idx/(24*60*7)))
    noise = np.random.normal(0, 2)
    return daily + weekly + noise

def inject_anomaly(value, kind):
    if kind == "spike": return value + np.random.uniform(50,120)
    if kind == "drop": return value - np.random.uniform(40,90)
    if kind == "drift": return value + np.random.uniform(20,60)
    if kind == "noise_burst": return value + np.random.normal(0,30)
    return value

def generate(n_series=12000, start_time=datetime(2025,6,1), freq_seconds=60, anomaly_fraction=0.03, out_csv='../data/synthetic_timeseries.csv'):
    rows = []
    anomaly_labels = np.zeros(n_series, dtype=int)
    for i in range(n_series):
        ts = start_time + timedelta(seconds=i*freq_seconds)
        base = base_load(i)
        voltage = 230 + np.random.normal(0, 0.5)
        current = base / voltage + np.random.normal(0, 0.02)
        power = voltage * current
        pf = max(0.7, min(1.0, 0.95 + np.random.normal(0, 0.02)))
        freq = 50 + np.random.normal(0, 0.02)
        harm = abs(np.random.normal(0, 0.01))
        if np.random.rand() < anomaly_fraction:
            kind = np.random.choice(["spike","drop","drift","noise_burst"])
            power = inject_anomaly(power, kind)
            current = power / voltage
            anomaly_labels[i] = 1
        rows.append({
            'timestamp': ts,
            'voltage': round(voltage,3),
            'current': round(current,4),
            'power': round(power,3),
            'pf': round(pf,3),
            'freq': round(freq,3),
            'harmonics': round(harm,5),
            'anomaly': int(anomaly_labels[i])
        })
    df = pd.DataFrame(rows)
    # features
    df['power_ma_5'] = df['power'].rolling(window=5, min_periods=1).mean()
    df['power_std_5'] = df['power'].rolling(window=5, min_periods=1).std().fillna(0)
    df['current_diff_1'] = df['current'].diff().fillna(0)
    df['power_pct_change_1'] = df['power'].pct_change().fillna(0)
    out_csv = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    # write meta
    meta = {
        'n_series': n_series,
        'anomaly_fraction': float(anomaly_fraction),
        'generated_at': str(datetime.now()),
        'csv_path': out_csv
    }
    meta_path = os.path.join(os.path.dirname(out_csv), 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset to {out_csv}")
    print(f"Saved meta to {meta_path}")
    return out_csv

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=12000)
    p.add_argument('--freq_seconds', type=int, default=60)
    p.add_argument('--anomaly_fraction', type=float, default=0.03)
    p.add_argument('--out', type=str, default='../data/synthetic_timeseries.csv')
    args = p.parse_args()
    generate(n_series=args.n, freq_seconds=args.freq_seconds, anomaly_fraction=args.anomaly_fraction, out_csv=args.out)
