#!/usr/bin/env python3
"""
Compare MinGBM with LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import time

# ANSI Color codes
CYAN = '\033[1;36m'
YELLOW = '\033[1;33m'
GREEN = '\033[1;32m'
RED = '\033[1;31m'
RESET = '\033[0m'

def train_lightgbm(csv_file, n_trees=50, learning_rate=0.1):
    """Train LightGBM model"""
    print("=== LightGBM Training ===\n")
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Separate features and target
    X = df.drop(['Id', 'SalePrice'], axis=1)
    y = df['SalePrice']
    
    # Convert categorical to numeric (label encoding)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"{GREEN}  ✓ Dataset: {X.shape[0]} samples, {X.shape[1]} features{RESET}")
    print(f"{GREEN}  ✓ Base score: {y.mean():.2f}{RESET}\n")
    
    # Train LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': learning_rate,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'verbose': -1,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'num_threads': 1
    }
    
    train_data = lgb.Dataset(X, label=y)
    
    print(f"{YELLOW}Training with {n_trees} trees, learning_rate={learning_rate}...{RESET}")
    start = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_trees,
        valid_sets=[train_data],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(period=10)]
    )
    
    elapsed = time.time() - start
    print(f"\n{GREEN}Training completed in {elapsed:.2f} seconds{RESET}")
    
    # Evaluate
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\n{CYAN}Final Metrics:{RESET}")
    print(f"{GREEN}  RMSE: {rmse:.2f}{RESET}")
    print(f"{GREEN}  MAE:  {mae:.2f}{RESET}")
    
    return rmse, mae, elapsed

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print(f"{YELLOW}Usage:{RESET} python compare_lgbm.py <csv_file> [n_trees] [learning_rate]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    n_trees = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    train_lightgbm(csv_file, n_trees, learning_rate)
