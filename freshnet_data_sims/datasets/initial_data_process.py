"""This script reads FreshRetailNet parquet data, cleans and processes it to extract modified demand and censor indicators, and outputs cleaned CSV files for both train and eval splits."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from typing import List, Sequence, Optional, Union


def get_modified_demand_and_censor_indicator(row):
    """Compute the minimal starting inventory (s_t) and censoring indicator from stock and sales data in each row."""
    # Parse status
    status = row["hours_stock_status"]
    if isinstance(status, str):
        status = ast.literal_eval(status)
    status = [int(bool(x)) for x in status]


    if status[0] == 1: # product initially out of stock
        s_t = 0
        cens = -1 # label it as DO NOT USE!
        return s_t, cens

    # Parse hourly sales if present; otherwise fall back to uniform over in-stock hours
    hourly_sales = row.get("hours_sale", None)
    if isinstance(hourly_sales, str):
        hourly_sales = ast.literal_eval(hourly_sales)

    # First OUT-OF-STOCK hour
    first_oos = next((h for h, v in enumerate(status) if v == 1), None)

    if first_oos is None:
        # No stockout → minimal start inventory is total sales
        s_t = max(row["sale_amount"],sum(hourly_sales))
        cens = 0
        s_t = round(s_t, 1)
    else:
        # Minimal start inventory = sales before first stockout
        s_t = sum(hourly_sales[:first_oos])
        cens = 1
        s_t = round(s_t, 1)

    return s_t, cens



def process_data_to_csv(file):
    """Load a parquet file, apply the transformation, drop irrelevant columns, filter out invalid rows, and save the processed data to CSV."""
    df = pd.read_parquet(f'{file}.parquet')


    df = df.drop(columns=['discount', 'holiday_flag', 'activity_flag', 'precpt',
                        'avg_temperature', 'avg_humidity', 'avg_wind_level'])
    # Dropping unnecessary columns
    df[["s_t", "censored"]] = df.apply(get_modified_demand_and_censor_indicator, axis=1, result_type="expand")


    print(f"Total rows: {len(df):,}")

    df_retained = df[df["censored"] != -1] # filters out the "unidentified" data

    print(f"Retained rows: {len(df_retained):,} ({100 * len(df_retained)/len(df):.2f}% retained)")
    print(f"Dropped rows: {len(df) - len(df_retained):,} ({100 * (len(df) - len(df_retained))/len(df):.2f}% dropped)")
    df_retained.to_csv(f'./{file}.csv', index = True)

process_data_to_csv('train')
process_data_to_csv('eval')