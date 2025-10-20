#!/usr/bin/env python3
"""
Check what formatting methods KNN has available
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

print("=== KNN FORMATTING METHODS ===")
knn = transactions_modeler.model

print("1. str(knn):")
print(str(knn))
print()

print("2. repr(knn):")
print(repr(knn))
print()

print("3. knn.__str__():")
print(knn.__str__())
print()

print("4. knn.__repr__():")
print(knn.__repr__())
print()

print("5. knn.get_params():")
print(knn.get_params())
print()

print("=== COMPARISON WITH OTHER MODELS ===")
print("\nLogisticRegression:")
lr = LogisticRegression()
print(f"str: {str(lr)}")
print(f"repr: {repr(lr)}")

print("\nRandomForestClassifier:")
rf = RandomForestClassifier()
print(f"str: {str(rf)}")
print(f"repr: {repr(rf)}")

print("\n=== BEST OPTION FOR KNN ===")
print("KNN doesn't have a nice built-in summary like other models")
print("But we can use str(knn) which gives: KNeighborsClassifier(n_neighbors=5, weights='distance')")
print("Or we can create our own custom summary")
