"""
Lab 2 — Data Pipeline: Retail Sales Analysis
Module 2 — Programming for AI & Data Science

Complete each function below. Remove the TODO: comments and pass statements
as you implement each function. Do not change the function signatures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = 'data/sales_records.csv'
OUTPUT_DIR = 'output'


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
    """Load sales records from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw sales records DataFrame.
    """
    
    df = pd.read_csv(filepath) 
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def clean_data(df):
    """Handle missing values and fix data types.

    - Fill missing 'quantity' values with the column median.
    - Fill missing 'unit_price' values with the column median.
    - Parse the 'date' column to datetime (use errors='coerce' to handle malformatted dates).
    - Print a progress message showing the record count after cleaning.

    Args:
        df (pd.DataFrame): Raw DataFrame from load_data().

    Returns:
        pd.DataFrame: Cleaned DataFrame (do not modify the input in place).
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return df
    df_clean = df.copy()
    for col in ['quantity', 'unit_price']:
        if df_clean[col].isnull().all():
            df_clean[col] = 0
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['quantity', 'unit_price'], how='all')
    print(f"Cleaned data: {len(df_clean)} records")

    return df_clean


def add_features(df):
    """Compute derived columns.

    - Add 'revenue' column: quantity * unit_price.
    - Add 'day_of_week' column: day name from the date column.

    Args:
        df (pd.DataFrame): Cleaned DataFrame from clean_data().

    Returns:
        pd.DataFrame: DataFrame with new columns added.
    """
    
    df_enriched = df.copy()
    df_enriched['revenue'] = df_enriched['quantity'] * df_enriched['unit_price']
    df_enriched['day_of_week']= df_enriched['date'].dt.day_name()
    return df_enriched


def generate_summary(df):
    """Compute summary statistics.

    Args:
        df (pd.DataFrame): Enriched DataFrame from add_features().

    Returns:
        dict: Summary with keys:
            - 'total_revenue': total revenue (sum)
            - 'avg_order_value': average order value (mean)
            - 'top_category': product category with highest total revenue
            - 'record_count': number of records in df
    """
    if df.empty:
        return {
            'total_revenue': 0,
            'avg_order_value': 0,
            'top_category': "N/A",
            'record_count': 0
        }
    
    total_rev = df['revenue'].sum()
    avg_order = df['revenue'].mean()
    count = len(df)
    if total_rev > 0:
        top_cat = df.groupby('product_category')['revenue'].sum().idxmax()
    else:
        top_cat = "N/A"
    summary_dict = {
        'total_revenue': total_rev,
        'avg_order_value': avg_order,
        'top_category': top_cat,
        'record_count': count
    }
    return summary_dict


def create_visualizations(df, output_dir=OUTPUT_DIR):
    """Create and save 3 charts as PNG files.

    Charts to create:
    1. Bar chart: total revenue by product category
    2. Line chart: daily revenue trend (aggregate revenue by date)
    3. Horizontal bar chart: average order value by payment method

    Save each chart as a PNG using fig.savefig().
    Do NOT use plt.show() — it blocks execution in pipeline scripts.
    Close each figure with plt.close(fig) after saving.

    Args:
        df (pd.DataFrame): Enriched DataFrame from add_features().
        output_dir (str): Directory to save PNG files (create if needed).
    """
    # TODO: Create the output directory: os.makedirs(output_dir, exist_ok=True)

    # TODO: Chart 1 — Bar chart: total revenue by product category
    #   - Group by 'product_category', sum 'revenue'
    #   - fig, ax = plt.subplots(figsize=(10, 6))
    #   - ax.bar(categories, values) or use ax.barh() for horizontal
    #   - Set title, labels
    #   - fig.savefig(f'{output_dir}/revenue_by_category.png', dpi=150, bbox_inches='tight')
    #   - plt.close(fig)

    # TODO: Chart 2 — Line chart: daily revenue trend
    #   - Group by 'date', sum 'revenue' — sort by date
    #   - ax.plot(dates, revenues)
    #   - fig.savefig(f'{output_dir}/daily_revenue_trend.png', ...)
    #   - plt.close(fig)

    # TODO: Chart 3 — Horizontal bar chart: avg order value by payment method
    #   - Group by 'payment_method', mean 'revenue'
    #   - ax.barh(methods, avg_values)
    #   - fig.savefig(f'{output_dir}/avg_order_by_payment.png', ...)
    #   - plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    # --- Chart 1: Bar chart — total revenue by product category ---
    cat_data = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(cat_data.index, cat_data.values, color='skyblue')
    ax.set_title('Total Revenue by Product Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Total Revenue ($)')
    plt.xticks(rotation=45)
    fig.savefig(f'{output_dir}/revenue_by_category.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    # --- Chart 2: Line chart — daily revenue trend ---
    daily_data = df.groupby('date')['revenue'].sum().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_data.index, daily_data.values, marker='o', linestyle='-', color='green')
    ax.set_title('Daily Revenue Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Revenue ($)')
    fig.savefig(f'{output_dir}/daily_revenue_trend.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    # --- Chart 3: Horizontal bar chart — average order value by payment method ---
    pay_data = df.groupby('payment_method')['revenue'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pay_data.index, pay_data.values, color='salmon')
    ax.set_title('Average Order Value by Payment Method')
    ax.set_xlabel('Average Revenue ($)')
    ax.set_ylabel('Payment Method')
    fig.savefig(f'{output_dir}/avg_order_by_payment.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f" completed ")


def main():
    """Run the full data pipeline end-to-end."""
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_features(df)
    summary = generate_summary(df)
    print("\n--- Summary ---")
    print(f"Total Revenue: ${summary['total_revenue']:,.2f}")
    print(f"Top Category: {summary['top_category']}")

    create_visualizations(df, output_dir=OUTPUT_DIR)
    print("Pipeline complete.  All steps executed successfully.")


if __name__ == "__main__":
     main()
