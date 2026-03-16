import numpy as np
import pandas as pd


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import algorithms
import helper

import freshnet_helper

import matplotlib.pyplot as plt
import seaborn as sns

'''
Generates table of identifiability rates on average over all products for different values of b

'''

df = pd.read_csv('./data/freshnet_identifiable.csv')

print(df.head(5))

ident_df = df[df['metric'] == 'identifiable']

# Number of products
total_products = ident_df['product'].nunique()
print(f"Total number of products: {total_products}")

# Percentage identifiable by (b,h)
grouped = ident_df.groupby(['b', 'h'])['value'].mean().reset_index()
grouped['pct_identifiable'] = grouped['value'] * 100
print("\nPercentage of identifiable products by (b, h):")
print(grouped[['b', 'h', 'pct_identifiable']])