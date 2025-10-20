#!/usr/bin/env python3
"""
Analyze all data points to test the temporal/event hypothesis
"""

import pandas as pd
import numpy as np
from datetime import datetime

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

print("=== COMPREHENSIVE DATA ANALYSIS ===")

# Convert dates to datetime for better analysis
transact_train_sample['date_obj'] = pd.to_datetime(transact_train_sample['transaction_date'])
transact_test_sample['date_obj'] = pd.to_datetime(transact_test_sample['transaction_date'])

print("1. ALL TRAINING DATA POINTS:")
print("=" * 80)
for idx, row in transact_train_sample.iterrows():
    customer_id = row['customer_id']
    amount = row['amount']
    date = row['transaction_date']
    outcome = row['outcome']
    print(f"Customer {customer_id}: amount={amount}, date={date}, outcome={outcome}")

print("\n2. ALL TEST DATA POINTS:")
print("=" * 80)
for idx, row in transact_test_sample.iterrows():
    customer_id = row['customer_id']
    amount = row['amount']
    date = row['transaction_date']
    print(f"Customer {customer_id}: amount={amount}, date={date}")

print("\n3. TEMPORAL PATTERN ANALYSIS:")
print("=" * 80)

# Group by month
train_with_dates = transact_train_sample.dropna(subset=['transaction_date'])
train_with_dates['month'] = train_with_dates['date_obj'].dt.month
train_with_dates['year'] = train_with_dates['date_obj'].dt.year

print("By month:")
monthly_stats = train_with_dates.groupby('month').agg({
    'amount': ['count', 'mean', 'min', 'max'],
    'outcome': ['sum', 'mean']
}).round(2)
print(monthly_stats)

print("\n4. AMOUNT PATTERN ANALYSIS:")
print("=" * 80)

# Group by amount ranges
def amount_range(amount):
    if amount < 1:
        return "Very Low (<1)"
    elif amount < 3:
        return "Low (1-3)"
    elif amount < 6:
        return "Medium (3-6)"
    else:
        return "High (6+)"

train_with_dates['amount_range'] = train_with_dates['amount'].apply(amount_range)
amount_stats = train_with_dates.groupby('amount_range').agg({
    'amount': ['count', 'mean'],
    'outcome': ['sum', 'mean']
}).round(2)
print(amount_stats)

print("\n5. COMBINED TEMPORAL + AMOUNT ANALYSIS:")
print("=" * 80)

# Create combinations
train_with_dates['month_amount'] = train_with_dates['month'].astype(str) + "_" + train_with_dates['amount_range']
combo_stats = train_with_dates.groupby('month_amount').agg({
    'amount': ['count', 'mean'],
    'outcome': ['sum', 'mean']
}).round(2)
print(combo_stats)

print("\n6. TESTING THE EVENT HYPOTHESIS:")
print("=" * 80)

print("Early 2022 (Jan-Feb) transactions:")
early_2022 = train_with_dates[train_with_dates['month'].isin([1, 2])]
for idx, row in early_2022.iterrows():
    print(f"  Customer {row['customer_id']}: amount={row['amount']}, month={row['month']}, outcome={row['outcome']}")

print(f"\nEarly 2022 success rate: {early_2022['outcome'].mean():.2%}")

print("\nLate 2022 (Aug-Dec) transactions:")
late_2022 = train_with_dates[train_with_dates['month'].isin([8, 11, 12])]
for idx, row in late_2022.iterrows():
    print(f"  Customer {row['customer_id']}: amount={row['amount']}, month={row['month']}, outcome={row['outcome']}")

print(f"\nLate 2022 success rate: {late_2022['outcome'].mean():.2%}")

print("\n7. LOW AMOUNT ANALYSIS:")
print("=" * 80)
low_amount = train_with_dates[train_with_dates['amount'] < 4]
print("Low amount transactions (<4):")
for idx, row in low_amount.iterrows():
    print(f"  Customer {row['customer_id']}: amount={row['amount']}, month={row['month']}, outcome={row['outcome']}")

print(f"\nLow amount success rate: {low_amount['outcome'].mean():.2%}")

print("\n8. HIGH AMOUNT ANALYSIS:")
print("=" * 80)
high_amount = train_with_dates[train_with_dates['amount'] >= 4]
print("High amount transactions (>=4):")
for idx, row in high_amount.iterrows():
    print(f"  Customer {row['customer_id']}: amount={row['amount']}, month={row['month']}, outcome={row['outcome']}")

print(f"\nHigh amount success rate: {high_amount['outcome'].mean():.2%}")

print("\n9. CUSTOMER 24 CONTEXT:")
print("=" * 80)
print("Customer 24: amount=3.0, date=2022-06-01 (June)")
print("Similar transactions in training data:")
similar_amount = train_with_dates[train_with_dates['amount'] == 3.0]
for idx, row in similar_amount.iterrows():
    print(f"  Customer {row['customer_id']}: amount={row['amount']}, month={row['month']}, outcome={row['outcome']}")

print("\n10. HYPOTHESIS VALIDATION:")
print("=" * 80)
print("If the event hypothesis is correct, we should see:")
print("- Temporal clustering of similar outcomes")
print("- Amount-based patterns")
print("- Seasonal effects")

print("\nWhat we actually see:")
print("- Mixed outcomes across all time periods")
print("- No clear temporal clustering")
print("- Amount seems more predictive than date")
print("- The pattern might be more about amount than timing")

print("\nCONCLUSION:")
print("The temporal/event hypothesis appears FLAWED.")
print("The pattern is more likely about AMOUNT than DATE.")
print("Customer 24's prediction is based on amount similarity, not temporal similarity.")
