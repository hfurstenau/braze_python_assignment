#!/usr/bin/env python3
"""
Threshold Analysis for Customer 24 Misclassification
"""

import pandas as pd
import numpy as np
from main import DataModeler

# Create the same data as in main.py
transact_train_sample = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            '2022-01-01',
            '2022-08-01',
            None,
            '2022-12-01',
            '2022-02-01',
            None,
            '2022-02-01',
            '2022-01-01',
            '2022-11-01',
            '2022-01-01'
        ],
        "outcome" : [False, True, True, True, False, False, True, True, True, False]
    }
)

transact_test_sample = pd.DataFrame(
    {
        "customer_id": [21, 22, 23, 24, 25],
        "amount": [0.5, np.nan, 8, 3, 2],
        "transaction_date": [
            '2022-02-01',
            '2022-11-01',
            '2022-06-01',
            None,
            '2022-02-01'
        ]
    }
)

print("=== THRESHOLD ANALYSIS FOR CUSTOMER 24 ===")
print("Customer 24: amount=3.0, date=2022-06-01 (mean) → Predicted: True, Expected: False")

# Initialize modeler
transactions_modeler = DataModeler(transact_train_sample)
transactions_modeler.prepare_data()
transactions_modeler.impute_missing()
transactions_modeler.fit()

# Prepare test data
adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)
filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)

print("\n1. PROBABILITY ANALYSIS:")
# Get probabilities for all test cases
probabilities = transactions_modeler.model.predict_proba(filled_test_sample[['amount', 'transaction_date']])
print("Test case probabilities (True, False):")
customer_ids = [21, 22, 23, 24, 25]
for i, (customer_id, prob) in enumerate(zip(customer_ids, probabilities)):
    print(f"  Customer {customer_id}: {prob[1]:.4f} (True), {prob[0]:.4f} (False)")

print("\n2. CUSTOMER 24 DETAILED ANALYSIS:")
customer_24_idx = 3  # Customer 24 is at index 3
customer_24_prob = probabilities[customer_24_idx]
print(f"Customer 24 probability of True: {customer_24_prob[1]:.4f}")
print(f"Customer 24 probability of False: {customer_24_prob[0]:.4f}")
print(f"Confidence: {max(customer_24_prob):.4f}")

print("\n3. THRESHOLD TUNING:")
# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in thresholds:
    predictions = probabilities[:, 1] > threshold
    accuracy = sum(predictions == [False, True, True, False, False]) / len(predictions) * 100
    print(f"  Threshold {threshold}: Accuracy = {accuracy:.1f}%")
    print(f"    Customer 24 prediction: {predictions[3]} (prob={customer_24_prob[1]:.4f})")

print("\n4. FEATURE IMPORTANCE:")
if hasattr(transactions_modeler.model, 'feature_importances_'):
    feature_names = ['amount', 'transaction_date']
    importances = transactions_modeler.model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f}")

print("\n5. DECISION TREE ANALYSIS:")
# Get the first tree to understand the decision path
if hasattr(transactions_modeler.model, 'estimators_'):
    first_tree = transactions_modeler.model.estimators_[0]
    print("First tree structure:")
    print(f"  Tree depth: {first_tree.get_depth()}")
    print(f"  Number of leaves: {first_tree.get_n_leaves()}")

print("\n6. CUSTOMER 24 FEATURE VALUES:")
customer_24_features = filled_test_sample.iloc[3][['amount', 'transaction_date']]
print(f"  Amount: {customer_24_features['amount']}")
print(f"  Transaction Date: {customer_24_features['transaction_date']:.0f}")

print("\n7. TRAINING DATA CONTEXT:")
# Find similar cases in training data
amount_3_cases = transactions_modeler.train_df[transactions_modeler.train_df['amount'] == 3.0]
print("Training cases with amount=3.0:")
for idx, row in amount_3_cases.iterrows():
    outcome = transactions_modeler.original_df.iloc[idx]['outcome']
    print(f"  Amount: {row['amount']}, Date: {row['transaction_date']:.0f}, Outcome: {outcome}")

print("\n8. ALTERNATIVE APPROACHES:")
print("  a) Try different Random Forest parameters")
print("  b) Use ensemble of multiple models")
print("  c) Add feature engineering (amount*date interaction)")
print("  d) Use different threshold based on training data distribution")
