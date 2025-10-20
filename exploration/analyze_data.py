import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import DataModeler

# Create the training data
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

# Create the test data
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

print("=== DATA ANALYSIS ===")
print("\n1. Original Training Data:")
print(transact_train_sample)
print(f"\nTraining data dtypes:\n{transact_train_sample.dtypes}")

print("\n2. Original Test Data:")
print(transact_test_sample)

# Process the data
transactions_modeler = DataModeler(transact_train_sample)
transactions_modeler.prepare_data()
transactions_modeler.impute_missing()
transactions_modeler.fit()

print("\n3. Processed Training Data:")
print(transactions_modeler.train_df)
print(f"\nProcessed dtypes:\n{transactions_modeler.train_df.dtypes}")

# Add outcome to processed data for analysis
processed_train = transactions_modeler.train_df.copy()
processed_train['outcome'] = transact_train_sample['outcome']

print("\n4. Correlation Analysis:")
print("Correlation matrix:")
correlation_matrix = processed_train[['amount', 'transaction_date', 'outcome']].corr()
print(correlation_matrix)

print("\n5. Feature Analysis:")
print("Amount statistics:")
print(f"  Mean: {processed_train['amount'].mean():.4f}")
print(f"  Std: {processed_train['amount'].std():.4f}")
print(f"  Min: {processed_train['amount'].min():.4f}")
print(f"  Max: {processed_train['amount'].max():.4f}")

print("\nTransaction date statistics:")
print(f"  Mean: {processed_train['transaction_date'].mean():.0f}")
print(f"  Std: {processed_train['transaction_date'].std():.0f}")
print(f"  Min: {processed_train['transaction_date'].min():.0f}")
print(f"  Max: {processed_train['transaction_date'].max():.0f}")

print("\n6. Outcome Analysis:")
print("Outcome distribution:")
print(processed_train['outcome'].value_counts())
print(f"Outcome rate: {processed_train['outcome'].mean():.2%}")

print("\n7. Amount vs Outcome Analysis:")
for outcome in [False, True]:
    subset = processed_train[processed_train['outcome'] == outcome]
    print(f"  Outcome {outcome}:")
    print(f"    Count: {len(subset)}")
    print(f"    Amount mean: {subset['amount'].mean():.4f}")
    print(f"    Amount std: {subset['amount'].std():.4f}")

print("\n8. Date vs Outcome Analysis:")
for outcome in [False, True]:
    subset = processed_train[processed_train['outcome'] == outcome]
    print(f"  Outcome {outcome}:")
    print(f"    Count: {len(subset)}")
    print(f"    Date mean: {subset['transaction_date'].mean():.0f}")
    print(f"    Date std: {subset['transaction_date'].std():.0f}")

# Test predictions
print("\n9. Test Predictions:")
adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)
filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)
test_predictions = transactions_modeler.predict(filled_test_sample)

print("Test data:")
print(filled_test_sample)
print(f"\nTest predictions: {test_predictions.tolist()}")
print(f"Expected predictions: [False, True, True, False, False]")

# Calculate accuracy
expected_predictions = [False, True, True, False, False]
accuracy = sum(test_predictions == expected_predictions) / len(expected_predictions) * 100
print(f"Accuracy: {accuracy:.1f}%")

print("\n9.1. Detailed Analysis of Wrong Prediction:")
print("Customer 24 (WRONG): amount=3.0, date=2022-06-01 (mean)")
print("  - Predicted: True, Expected: False")
print("  - This is the only misclassified case")

print("\n9.2. Comparison with Training Data:")
print("Training data with amount=3.0:")
amount_3_cases = processed_train[processed_train['amount'] == 3.0]
print(amount_3_cases[['amount', 'transaction_date']])
print("Outcomes for amount=3.0:", processed_train[processed_train['amount'] == 3.0]['outcome'].tolist())

print("\n9.3. Training data with similar date (2022-06-01):")
# Convert test date to compare
test_date_float = pd.to_datetime('2022-06-01').value  # nanoseconds since epoch
date_tolerance = 30 * 24 * 60 * 60 * 1000000000  # 30 days in nanoseconds
similar_dates = processed_train[
    abs(processed_train['transaction_date'] - test_date_float) < date_tolerance
]
print(similar_dates[['amount', 'transaction_date']])
print("Outcomes for similar dates:", similar_dates['outcome'].tolist())

print("\n9.4. Feature Analysis for Customer 24:")
customer_24_amount = 3.0
customer_24_date = test_date_float
print(f"  Amount: {customer_24_amount}")
print(f"  Date: {customer_24_date:.0f} (2022-06-01)")

print("\n9.5. Training Data Statistics for Context:")
print("Amount statistics for False outcomes:")
false_amounts = processed_train[processed_train['outcome'] == False]['amount']
print(f"  Mean: {false_amounts.mean():.4f}")
print(f"  Std: {false_amounts.std():.4f}")
print(f"  Min: {false_amounts.min():.4f}")
print(f"  Max: {false_amounts.max():.4f}")

print("Amount statistics for True outcomes:")
true_amounts = processed_train[processed_train['outcome'] == True]['amount']
print(f"  Mean: {true_amounts.mean():.4f}")
print(f"  Std: {true_amounts.std():.4f}")
print(f"  Min: {true_amounts.min():.4f}")
print(f"  Max: {true_amounts.max():.4f}")

print("\n10. Feature Importance (if available):")
if hasattr(transactions_modeler.model, 'feature_importances_'):
    feature_names = ['amount', 'transaction_date']
    importances = transactions_modeler.model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f}")

print("\n11. Model Summary:")
print(transactions_modeler.model_summary())

print("\n12. Creating scatter plot visualization...")

# Create scatter plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Amount vs Outcome
colors = ['red' if outcome else 'blue' for outcome in processed_train['outcome']]
ax1.scatter(processed_train['amount'], processed_train['outcome'], c=colors, alpha=0.7, s=100, label='Training Data')
ax1.set_xlabel('Amount')
ax1.set_ylabel('Outcome')
ax1.set_title('Amount vs Outcome')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['False', 'True'])
ax1.grid(True, alpha=0.3)

# Add test data points
test_colors = ['red' if pred else 'blue' for pred in test_predictions]
ax1.scatter(filled_test_sample['amount'], [0.5]*len(filled_test_sample), 
           c=test_colors, alpha=0.7, s=100, marker='s', label='Test Data (Predictions)')

# Add ground truth for test data
expected_predictions = [False, True, True, False, False]
ground_truth_colors = ['red' if outcome else 'blue' for outcome in expected_predictions]
ax1.scatter(filled_test_sample['amount'], [0.3]*len(filled_test_sample), 
           c=ground_truth_colors, alpha=0.7, s=100, marker='^', label='Test Data (Ground Truth)')

# Create comprehensive legend with colors and data types
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='True Outcome'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='False Outcome'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Training Data'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='Test Data (Predictions)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Test Data (Ground Truth)')
]
ax1.legend(handles=legend_elements, loc='upper right')

# Plot 2: Transaction Date vs Outcome (scaled for better visualization)
# Convert timestamps to days since epoch for better visualization
train_dates_days = (processed_train['transaction_date'] - 1640995200000000000) / (24 * 60 * 60 * 1000000000)
test_dates_days = (filled_test_sample['transaction_date'] - 1640995200000000000) / (24 * 60 * 60 * 1000000000)

ax2.scatter(train_dates_days, processed_train['outcome'], c=colors, alpha=0.7, s=100)
ax2.scatter(test_dates_days, [0.5]*len(filled_test_sample), 
           c=test_colors, alpha=0.7, s=100, marker='s', label='Test Data (Predictions)')
ax2.scatter(test_dates_days, [0.3]*len(filled_test_sample), 
           c=ground_truth_colors, alpha=0.7, s=100, marker='^', label='Test Data (Ground Truth)')
ax2.set_xlabel('Days since 2022-01-01')
ax2.set_ylabel('Outcome')
ax2.set_title('Transaction Date vs Outcome')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['False', 'True'])
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Amount vs Transaction Date (2D feature space)
ax3.scatter(processed_train['amount'], train_dates_days, c=colors, alpha=0.7, s=100, label='Training Data')
ax3.scatter(filled_test_sample['amount'], test_dates_days, c=test_colors, alpha=0.7, s=100, marker='s', label='Test Data (Predictions)')
ax3.scatter(filled_test_sample['amount'], test_dates_days, c=ground_truth_colors, alpha=0.7, s=100, marker='^', label='Test Data (Ground Truth)')
ax3.set_xlabel('Amount')
ax3.set_ylabel('Days since 2022-01-01')
ax3.set_title('Amount vs Transaction Date')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Transaction Date vs Amount (with human-readable dates)
# Convert timestamps to datetime objects for better x-axis labels
train_dates_datetime = pd.to_datetime(processed_train['transaction_date'])
test_dates_datetime = pd.to_datetime(filled_test_sample['transaction_date'])

ax4.scatter(train_dates_datetime, processed_train['amount'], c=colors, alpha=0.7, s=100, label='Training Data')
ax4.scatter(test_dates_datetime, filled_test_sample['amount'], c=test_colors, alpha=0.7, s=100, marker='s', label='Test Data (Predictions)')
ax4.scatter(test_dates_datetime, filled_test_sample['amount'], c=ground_truth_colors, alpha=0.7, s=100, marker='^', label='Test Data (Ground Truth)')
ax4.set_xlabel('Transaction Date')
ax4.set_ylabel('Amount')
ax4.set_title('Transaction Date vs Amount')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Format x-axis to show dates nicely
import matplotlib.dates as mdates
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax4.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved as 'data_visualization.png'")

# Show the plot
plt.show()

print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*50)

# Calculate training accuracy
train_predictions = transactions_modeler.predict()
train_accuracy = sum(train_predictions == processed_train['outcome']) / len(processed_train) * 100
print(f"Training Accuracy: {train_accuracy:.1f}%")

# Calculate test accuracy
test_accuracy = sum(test_predictions == expected_predictions) / len(expected_predictions) * 100
print(f"Test Accuracy: {test_accuracy:.1f}%")

