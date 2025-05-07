import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# Potentially define dtypes for memory efficiency (inspect your data for appropriate types)
# dtype_dict = {
#     'OrderId': 'int32',
#     'TransactionId': 'object',
#     # ... other columns
#     'OrderQty': 'int16',
#     'CanceledQty': 'int16',
#     'MenuPrice': 'float32', # After cleaning
#     'MenuSubsidy': 'float32' # After cleani# }

try:
    # df = pd.read_csv('your_data.csv', dtype=dtype_dict) # If using dtype_dict
    address = 'venvx/AnnonymData.csv'

    df = pd.read_csv(address)
except MemoryError:
    print("MemoryError: Consider loading data in chunks or using a more memory-efficient library if pandas struggles with the full load at once.")
    # Add chunking logic or alternative library (Dask, Polars) if needed

print("Data loaded. Info:")
df.info(verbose=True, show_counts=True)
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe(include='all'))