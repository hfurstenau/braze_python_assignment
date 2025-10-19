import pytest
import pandas as pd
import numpy as np
from main import DataModeler


@pytest.fixture
def transact_train_sample():
    return pd.DataFrame(
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


@pytest.fixture
def transact_test_sample():
    return pd.DataFrame(
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


def test_training_sample(transact_train_sample):
    # Expected output from lines 141-151
    expected_df = pd.DataFrame({
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1.0, 3.0, 12.0, 6.0, 0.5, 0.2, np.nan, 5.0, np.nan, 3.0],
        "transaction_date": ['2022-01-01', '2022-08-01', None, '2022-12-01', '2022-02-01', None, '2022-02-01', '2022-01-01', '2022-11-01', '2022-01-01'],
        "outcome": [False, True, True, True, False, False, True, True, True, False]
    })
    
    pd.testing.assert_frame_equal(transact_train_sample, expected_df)


def test_current_dtypes(transact_train_sample):
    # Expected dtypes from lines 158-162
    expected_dtypes = pd.Series({
        'customer_id': 'int64',
        'amount': 'float64', 
        'transaction_date': 'object',
        'outcome': 'bool'
    })
    
    pd.testing.assert_series_equal(transact_train_sample.dtypes, expected_dtypes)


def test_changed_columns_dtypes(transact_train_sample):
    transactions_modeler = DataModeler(transact_train_sample)
    transactions_modeler.prepare_data()
    
    # Expected dtypes from lines 173-174
    expected_dtypes = pd.Series({
        'amount': 'float64',
        'transaction_date': 'float64'
    })
    
    pd.testing.assert_series_equal(transactions_modeler.train_df.dtypes, expected_dtypes)


def test_imputed_missing_as_mean(transact_train_sample):
    transactions_modeler = DataModeler(transact_train_sample)
    transactions_modeler.prepare_data()
    transactions_modeler.impute_missing()
    
    # Expected output from train sample
    expected_amounts = [1.0000, 3.0000, 12.0000, 6.0000, 0.5000, 0.2000, 3.8375, 5.0000, 3.8375, 3.0000]
    expected_dates = [1.640995e+18, 1.659312e+18, 1.650845e+18, 1.669853e+18, 1.643674e+18, 1.650845e+18, 1.643674e+18, 1.640995e+18, 1.667261e+18, 1.640995e+18]
    
    np.testing.assert_array_almost_equal(transactions_modeler.train_df['amount'].values, expected_amounts, decimal=4)
    np.testing.assert_array_almost_equal(transactions_modeler.train_df['transaction_date'].values, expected_dates, decimal=6)


def test_accuracy_training(transact_train_sample):
    transactions_modeler = DataModeler(transact_train_sample)
    transactions_modeler.prepare_data()
    transactions_modeler.impute_missing()
    transactions_modeler.fit()
    
    in_sample_predictions = transactions_modeler.predict()
    expected_predictions = [False, True, True, True, False, False, True, True, True, False]
    accuracy = sum(in_sample_predictions == expected_predictions) / len(expected_predictions) * 100
    
    # Expected accuracy from line 212
    assert accuracy == 100.0


def test_test_sample_dtypes(transact_train_sample, transact_test_sample):
    transactions_modeler = DataModeler(transact_train_sample)
    transactions_modeler.prepare_data()
    transactions_modeler.impute_missing()
    transactions_modeler.fit()
    
    adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)
    
    # Expected dtypes from lines 243-245
    expected_dtypes = pd.Series({
        'amount': 'float64',
        'transaction_date': 'float64'
    })
    
    pd.testing.assert_series_equal(adjusted_test_sample.dtypes, expected_dtypes)


def test_test_sample_imputed_missing(transact_train_sample, transact_test_sample):
    transactions_modeler = DataModeler(transact_train_sample)
    transactions_modeler.prepare_data()
    transactions_modeler.impute_missing()
    transactions_modeler.fit()
    
    adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)
    filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)
    
    # Expected output from test sample
    expected_amounts = [0.5000, 3.8375, 8.0000, 3.0000, 2.0000]
    expected_dates = [1.643674e+18, 1.667261e+18, 1.654042e+18, 1.650845e+18, 1.643674e+18]
    
    np.testing.assert_array_almost_equal(filled_test_sample['amount'].values, expected_amounts, decimal=4)
    np.testing.assert_array_almost_equal(filled_test_sample['transaction_date'].values, expected_dates, decimal=6)


def test_out_of_sample_accuracy(transact_train_sample, transact_test_sample):
    transactions_modeler = DataModeler(transact_train_sample)
    transactions_modeler.prepare_data()
    transactions_modeler.impute_missing()
    transactions_modeler.fit()
    
    adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)
    filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)
    oos_predictions = transactions_modeler.predict(filled_test_sample)
    
    expected_predictions = [False, True, True, False, False]
    accuracy = sum(oos_predictions == expected_predictions) / len(expected_predictions) * 100
    
    # Expected accuracy from line 267
    assert accuracy == 100.0
