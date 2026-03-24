
"""train_model.py
Train IsolationForest (unsupervised) and RandomForest (supervised) on the generated dataset.
Saves models to ../models/
Usage:
    python train_model.py --data ../data/synthetic_timeseries.csv
"""
import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib

FEATURE_COLS = [
    'voltage','current','power','pf','freq','harmonics',
    'power_ma_5','power_std_5','current_diff_1','power_pct_change_1'
]

def train(data_csv='../data/synthetic_timeseries.csv', out_dir='../models'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_csv, parse_dates=['timestamp'])
    X = df[FEATURE_COLS].fillna(0).values
    y = df['anomaly'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    # Isolation Forest
    anomaly_fraction = max(0.01, float(y.mean()))
    iso = IsolationForest(contamination=anomaly_fraction, random_state=42, n_estimators=200)
    iso.fit(X_train)
    y_pred_iso = (iso.predict(X_test) == -1).astype(int)
    iso_prec = precision_score(y_test, y_pred_iso, zero_division=0)
    iso_rec = recall_score(y_test, y_pred_iso, zero_division=0)
    iso_f1 = f1_score(y_test, y_pred_iso, zero_division=0)
    joblib.dump(iso, os.path.join(out_dir, 'isolation_forest.joblib'))
    # Random Forest (supervised)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
    rf_rec = recall_score(y_test, y_pred_rf, zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
    joblib.dump(rf, os.path.join(out_dir, 'random_forest.joblib'))
    metrics = {
        'isolation_forest': {'precision': iso_prec, 'recall': iso_rec, 'f1': iso_f1},
        'random_forest': {'precision': rf_prec, 'recall': rf_rec, 'f1': rf_f1},
        'feature_cols': FEATURE_COLS
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Training complete. Models and metrics saved to', out_dir)
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='../data/synthetic_timeseries.csv')
    p.add_argument('--out', type=str, default='../models')
    args = p.parse_args()
    train(data_csv=args.data, out_dir=args.out)
