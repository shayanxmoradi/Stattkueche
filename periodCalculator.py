import pandas as pd

# === CONFIGURATION ===
file_path = "large_file.csv"      # Your 1GB CSV file
date_column = "timestamp"         # Replace with your actual column name

# === GET FIRST ROW ===
first_row = pd.read_csv(file_path, usecols=[date_column], nrows=1)
start_date = pd.to_datetime(first_row[date_column].iloc[0])

# === GET LAST ROW (FAST WAY) ===
# Use tail with chunks
last_date = None
chunk_size = 100_000

for chunk in pd.read_csv(file_path, usecols=[date_column], chunksize=chunk_size):
    last_date = pd.to_datetime(chunk[date_column].iloc[-1])

# === OUTPUT ===
print(f"🕓 Data period: {start_date} → {last_date}")
