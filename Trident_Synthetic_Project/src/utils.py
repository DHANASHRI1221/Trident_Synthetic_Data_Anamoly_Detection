
"""utils.py
Helper utilities used by scripts: loading data, simple plotting helpers, small evaluation helpers.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(path='../data/synthetic_timeseries.csv'):
    return pd.read_csv(path, parse_dates=['timestamp'])

def quick_plot_power(df, out_path=None, show=True):
    plt.figure(figsize=(12,4))
    plt.plot(df['timestamp'], df['power'])
    plt.title('Power Time-Series')
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()

def save_json(obj, path):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
