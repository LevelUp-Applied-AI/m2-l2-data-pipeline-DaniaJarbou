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
import json
import sys


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
   
    
    df = pd.read_csv(filepath) 
    print(f"Loaded {len(df)} records from {filepath}")
    return df

def clean_data(df, cleaning_config):
   
    if df.empty:
        print("Warning: DataFrame is empty")
        return df
    df_clean = df.copy()
    cols_to_fill = cleaning_config.get('fill_columns', [])
    strategy = cleaning_config.get('strategy', 'median')
    for col in cols_to_fill:
        if col in df_clean.columns:
            if df_clean[col].isnull().all():
                fill_val = 0
            elif strategy == 'mean':
                fill_val = df_clean[col].mean()
            else:
                fill_val = df_clean[col].median()
                
            df_clean[col] = df_clean[col].fillna(fill_val)

    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    print(f"Cleaned data: {len(df_clean)} records")
    return df_clean


def add_features(df):
   
    df_enriched = df.copy()
    if 'quantity' in df_enriched.columns and 'unit_price' in df_enriched.columns:
        df_enriched['revenue'] = df_enriched['quantity'] * df_enriched['unit_price']
    
    if 'date' in df_enriched.columns:
        df_enriched['day_of_week'] = df_enriched['date'].dt.day_name()
    return df_enriched


def generate_summary(df, group_by_col):
   
    if df.empty:
       return {'total_revenue': 0, 'avg_order_value': 0, 'top_category': "N/A", 'record_count': 0}
    
    total_rev = df['revenue'].sum() if 'revenue' in df.columns else 0
    avg_order = df['revenue'].mean() if 'revenue' in df.columns else 0
    count = len(df)
    if total_rev > 0 and group_by_col in df.columns:
        top_val = df.groupby(group_by_col)['revenue'].sum().idxmax()
    else:
        top_val = "N/A"
    summary_dict = {
        'total_revenue': total_rev,
        'avg_order_value': avg_order,
        'top_category': top_val,
        'record_count': count
    }
    return summary_dict


def create_visualizations(df, output_dir, charts_to_make):
    
    os.makedirs(output_dir, exist_ok=True)
    if "revenue_by_category" in charts_to_make and 'product_category' in df.columns:
        cat_data = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(cat_data.index, cat_data.values, color='skyblue')
        ax.set_title('Total Revenue by Product Category')
        fig.savefig(f'{output_dir}/revenue_by_category.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Visualizations completed in {output_dir}")
  


def main(config_path="config_sales.json"):
   if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return
   with open(config_path, 'r') as f:
        config = json.load(f)
   df = load_data(config['input_path'])
   df = clean_data(df, config['cleaning'])
   df = add_features(df)
   summary = generate_summary(df, config['group_by_col'])
   print(f"Summary for {config['input_path']}:", summary)
   create_visualizations(df, config['output_dir'], config.get('charts', []))
   if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    


if __name__ == "__main__":
     path = sys.argv[1] if len(sys.argv) > 1 else "config_sales.json"
     main(path)
