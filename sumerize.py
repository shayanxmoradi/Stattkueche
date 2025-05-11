import pandas as pd

# === CONFIGURATION ===
input_file = "venvx/annonymdata.csv"       # Your original 1GB CSV
output_file = "sample_output.csv"   # Your smaller output file
num_rows = 20                     # Number of rows to keep
# selected_columns = ['Column1', 'Column2', 'Column3']  # Edit this!

# === PROCESSING ===
# Use chunks to handle large file efficiently
chunk_size = 20  # Read 10,000 rows at a time
rows_collected = []

for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    rows_needed = num_rows - len(rows_collected)
    if rows_needed <= 0:
        break
    rows_collected.append(chunk.head(rows_needed))

# Concatenate all collected rows
small_df = pd.concat(rows_collected)

# Write to output CSV
small_df.to_csv(output_file, index=False)

print(f"✅ Sample CSV saved as '{output_file}' with {len(small_df)} rows.")
