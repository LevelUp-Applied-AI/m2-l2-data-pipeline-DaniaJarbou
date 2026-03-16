"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""

import pandas as pd
import numpy as np
import pytest
from pipeline import load_data, clean_data, add_features


# ─── Test 1 ───────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    """load_data should return a DataFrame with expected columns and rows."""
    df = load_data('data/sales_records.csv')
    assert isinstance(df, pd.DataFrame), "The result should be a pandas DataFrame"
    assert len(df) > 0, "The DataFrame should not be empty"
    expected_columns = ['date', 'store_id', 'product_category', 'quantity', 'unit_price', 'payment_method']
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' is missing from the DataFrame"
       


# ─── Test 2 ───────────────────────────────────────────────────────────────────

def test_clean_data_no_nulls():
    """After clean_data, quantity and unit_price should have no NaN values."""
    raw_df = load_data('data/sales_records.csv')
    cleaned = clean_data(raw_df)
    assert cleaned['quantity'].isna().sum() == 0, "Found NaN values in quantity after cleaning"
    assert cleaned['unit_price'].isna().sum() == 0, "Found NaN values in unit_price after cleaning"



# ─── Test 3 ───────────────────────────────────────────────────────────────────

def test_add_features_creates_revenue():
    """add_features should add a 'revenue' column equal to quantity * unit_price."""
    raw_df = load_data('data/sales_records.csv')
    cleaned_df = clean_data(raw_df)
    enriched_df = add_features(cleaned_df)
    assert 'revenue' in enriched_df.columns, "Column 'revenue' was not created"
    expected_revenue = enriched_df['quantity'] * enriched_df['unit_price']
    pd.testing.assert_series_equal(
        enriched_df['revenue'], 
        expected_revenue, 
        check_names= False,
        obj="Revenue calculation"
        )