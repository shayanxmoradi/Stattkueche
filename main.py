# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly # Using plotly for potentially interactive plots
import matplotlib.pyplot as plt # Fallback for non-interactive environments
import numpy as np
import streamlit as st

addres ='venvx/AnnonymData.csv'
#df = pd.read_csv('venvx/AnnonymData.csv')
# df = pd.read_csv('venvx/AnnonymData.csv', engine='python', nrows=100)
#
#
# print(df.head)
# print(df.isnull().sum())
#
# print(df.describe())
pd.set_option('display.max_columns', None)
df_preview = pd.read_csv(addres, nrows=1000)
# print(df_preview.head())
#
st.dataframe(df_preview)

# chunksize = 50000  # Adjust based on your RAM
# for chunk in pd.read_csv(addres, chunksize=chunksize):
#     print(chunk.head())  # Preview each chunk
#     break
