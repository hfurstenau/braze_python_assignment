#!/usr/bin/env python3
"""
Check KNeighborsClassifier attributes for summary
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from main import DataModeler

# Create the same data
transact_train_sample = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            '2022-01-01', '2022-08-01', None, '2022-12-01', '2022-02-01',
            None, '2022-02-01', '2022-01-01', '2022-11-01', '2022-01-01'
        ],
        "outcome" : [False, True, True, True, False, False, True, True, True, False]
    }
)

# Initialize modeler
transactions_modeler = DataModeler(transact_train_sample)
transactions_modeler.prepare_data()
transactions_modeler.impute_missing()
transactions_modeler.fit()

print("=== KNN MODEL ATTRIBUTES ===")
print(f"Model type: {type(transactions_modeler.model)}")
print(f"Model attributes: {dir(transactions_modeler.model)}")

print("\n=== USEFUL ATTRIBUTES ===")
print(f"n_neighbors: {transactions_modeler.model.n_neighbors}")
print(f"weights: {transactions_modeler.model.weights}")
print(f"algorithm: {transactions_modeler.model.algorithm}")
print(f"metric: {transactions_modeler.model.metric}")
print(f"metric_params: {transactions_modeler.model.metric_params}")

print("\n=== TRAINING DATA INFO ===")
print(f"Number of training samples: {len(transactions_modeler.train_df)}")
print(f"Number of features: {len(transactions_modeler.train_df.columns)}")
print(f"Feature names: {list(transactions_modeler.train_df.columns)}")

print("\n=== POSSIBLE SUMMARY OPTIONS ===")
print("1. Basic: 'KNeighborsClassifier with 2 features: amount and transaction_date'")
print("2. Detailed: Include n_neighbors and weights")
print("3. Full: Include all parameters")
print("4. Custom: Create a meaningful summary")
