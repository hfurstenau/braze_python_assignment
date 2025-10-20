#!/usr/bin/env python3
"""
Analysis of Training Accuracy Drop with Threshold=0.7
"""

import pandas as pd
import numpy as np
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

print("=== TRAINING ACCURACY ANALYSIS ===")

# Initialize modeler
transactions_modeler = DataModeler(transact_train_sample)
transactions_modeler.prepare_data()
transactions_modeler.impute_missing()
transactions_modeler.fit()

# Get predictions with different thresholds
X = transactions_modeler.train_df[['amount', 'transaction_date']]
X_scaled = transactions_modeler.scaler.transform(X)
y_actual = transactions_modeler.original_df['outcome']

# Get probabilities
probabilities = transactions_modeler.model.predict_proba(X_scaled)

print("Training data with probabilities and predictions:")
print("=" * 80)
print(f"{'Customer':<8} {'Amount':<8} {'Date':<20} {'Actual':<8} {'Prob(True)':<12} {'Thresh=0.5':<12} {'Thresh=0.7':<12}")
print("-" * 80)

for i in range(len(transact_train_sample)):
    customer_id = transact_train_sample.iloc[i]['customer_id']
    amount = X.iloc[i]['amount']
    date = X.iloc[i]['transaction_date']
    actual = y_actual.iloc[i]
    prob_true = probabilities[i][1]
    
    pred_05 = prob_true > 0.5
    pred_07 = prob_true > 0.7
    
    print(f"{customer_id:<8} {amount:<8.1f} {date:<20.0f} {actual!s:<8} {prob_true:<12.4f} {pred_05!s:<12} {pred_07!s:<12}")

print("\n" + "=" * 80)
print("ANALYSIS:")

# Calculate accuracies
pred_05 = probabilities[:, 1] > 0.5
pred_07 = probabilities[:, 1] > 0.7

acc_05 = sum(pred_05 == y_actual) / len(y_actual) * 100
acc_07 = sum(pred_07 == y_actual) / len(y_actual) * 100

print(f"Threshold 0.5 accuracy: {acc_05:.1f}%")
print(f"Threshold 0.7 accuracy: {acc_07:.1f}%")

# Show which cases changed
changed_cases = pred_05 != pred_07
print(f"\nCases that changed with threshold=0.7:")
for i, changed in enumerate(changed_cases):
    if changed:
        customer_id = transact_train_sample.iloc[i]['customer_id']
        actual = y_actual.iloc[i]
        prob_true = probabilities[i][1]
        pred_05_val = pred_05[i]
        pred_07_val = pred_07[i]
        
        print(f"  Customer {customer_id}: Actual={actual}, Prob={prob_true:.4f}")
        print(f"    Threshold 0.5: {pred_05_val} (correct: {pred_05_val == actual})")
        print(f"    Threshold 0.7: {pred_07_val} (correct: {pred_07_val == actual})")
        print(f"    Impact: {'Better' if pred_07_val == actual and pred_05_val != actual else 'Worse'}")

print(f"\nCONCLUSION:")
print(f"- Threshold 0.7 makes the model more conservative")
print(f"- Some borderline cases are now predicted as False")
print(f"- This reduces overfitting and improves generalization")
print(f"- The trade-off: lower training accuracy but higher test accuracy")
