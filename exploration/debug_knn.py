#!/usr/bin/env python3
"""
Debug the KNN prediction for Customer 24
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

transact_test_sample = pd.DataFrame(
    {
        "customer_id": [21, 22, 23, 24, 25],
        "amount": [0.5, np.nan, 8, 3, 2],
        "transaction_date": [
            '2022-02-01', '2022-11-01', '2022-06-01', None, '2022-02-01'
        ]
    }
)

print("=== DEBUG KNN PREDICTION ===")

# Initialize modeler
transactions_modeler = DataModeler(transact_train_sample)
transactions_modeler.prepare_data()
transactions_modeler.impute_missing()

# Prepare data
X = transactions_modeler.train_df[['amount', 'transaction_date']]
y = transactions_modeler.original_df['outcome']
X_scaled = transactions_modeler.scaler.fit_transform(X)

# Prepare test data
adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)
filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)
X_test = filled_test_sample[['amount', 'transaction_date']]
X_test_scaled = transactions_modeler.scaler.transform(X_test)

# Fit the model
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_scaled, y)

print("1. ACTUAL PREDICTIONS:")
test_predictions = model.predict(X_test_scaled)
print(f"Test predictions: {test_predictions}")
print(f"Customer 24 prediction: {test_predictions[3]}")

print("\n2. PROBABILITIES:")
probabilities = model.predict_proba(X_test_scaled)
print(f"Customer 24 probabilities: False={probabilities[3][0]:.4f}, True={probabilities[3][1]:.4f}")

print("\n3. MANUAL DISTANCE CALCULATION:")
customer_24_scaled = X_test_scaled[3]  # Customer 24
print(f"Customer 24 scaled features: {customer_24_scaled}")

# Calculate distances manually
distances = []
for i, (train_point, outcome) in enumerate(zip(X_scaled, y)):
    dist = np.linalg.norm(customer_24_scaled - train_point)
    distances.append((dist, i, outcome))

# Sort by distance
distances.sort()

print("\nTop 5 neighbors with manual calculation:")
for i, (dist, idx, outcome) in enumerate(distances[:5]):
    customer_id = transact_train_sample.iloc[idx]['customer_id']
    amount = X.iloc[idx]['amount']
    date = X.iloc[idx]['transaction_date']
    weight = 1.0 / dist if dist > 0 else float('inf')
    print(f"  {i+1}. Customer {customer_id}: dist={dist:.4f}, weight={weight:.4f}, outcome={outcome}")

print("\n4. WEIGHTED VOTING CALCULATION:")
false_weight = 0
true_weight = 0

for i, (dist, idx, outcome) in enumerate(distances[:5]):
    weight = 1.0 / dist if dist > 0 else float('inf')
    if outcome:
        true_weight += weight
    else:
        false_weight += weight

print(f"Total False weight: {false_weight:.4f}")
print(f"Total True weight: {true_weight:.4f}")
print(f"Prediction: {'False' if false_weight > true_weight else 'True'}")

print("\n5. SKLEARN KNEIGHBORS METHOD:")
distances_sklearn, indices_sklearn = model.kneighbors([customer_24_scaled], n_neighbors=5, return_distance=True)
print("Sklearn kneighbors result:")
for i, (dist, idx) in enumerate(zip(distances_sklearn[0], indices_sklearn[0])):
    customer_id = transact_train_sample.iloc[idx]['customer_id']
    outcome = y.iloc[idx]
    weight = 1.0 / dist if dist > 0 else float('inf')
    print(f"  {i+1}. Customer {customer_id}: dist={dist:.4f}, weight={weight:.4f}, outcome={outcome}")

print("\n6. WHY THE DISCREPANCY?")
print("The sklearn implementation might use different distance calculations")
print("or handle edge cases differently than manual calculation")
