import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assume your DataFrame 'df' is already loaded and preprocessed
# For example:
file_path = 'venvx/AnnonymData.csv'
df = pd.read_csv(file_path)
# # Ensure CanceledQty is numeric (it should be from previous scripts)
# df['CanceledQty'] = pd.to_numeric(df['CanceledQty'], errors='coerce').fillna(0)


# --- Create a more representative sample DataFrame for demonstration ---
# This sample mimics the highly skewed nature of CanceledQty
n_rows = 10000
data_sample = {
    'CanceledQty': [0] * int(n_rows * 0.85) +  # 85% are 0
                   [1] * int(n_rows * 0.10) +  # 10% are 1
                   [2] * int(n_rows * 0.03) +  # 3% are 2
                   np.random.randint(3, 20, size=int(n_rows * 0.01)).tolist() +  # 1% between 3 and 19
                   np.random.randint(20, 196, size=int(n_rows * 0.01)).tolist()  # 1% higher values up to 195
}
# Ensure the list has n_rows elements
if len(data_sample['CanceledQty']) < n_rows:
    data_sample['CanceledQty'].extend([0] * (n_rows - len(data_sample['CanceledQty'])))
elif len(data_sample['CanceledQty']) > n_rows:
    data_sample['CanceledQty'] = data_sample['CanceledQty'][:n_rows]

df_sample = pd.DataFrame(data_sample)
# Replace df_sample with your actual DataFrame 'df' in your code
# df = df_sample # For testing this script directly
# print(df['CanceledQty'].describe())
# print(df['CanceledQty'].value_counts().nlargest(5))


# --- Plotting CanceledQty Distribution ---
target_variable = 'CanceledQty'

if target_variable in df.columns and pd.api.types.is_numeric_dtype(df[target_variable]):
    print(f"Generating distribution plots for '{target_variable}'...")

    # Plot 1: Bar Plot of Value Counts for common cancellation quantities
    plt.figure(figsize=(12, 7))
    # Get the counts for the most common cancellation quantities (e.g., top 10 or specific values like 0, 1, 2, 3, 4, 5)
    # For very skewed data, focusing on the most frequent values is key.
    value_counts_data = df[target_variable].value_counts()

    # Filter for specific common values if the tail is too long, e.g., 0 to 5
    common_values_to_plot = value_counts_data[
        value_counts_data.index <= 5].sort_index()  # Plotting counts for 0,1,2,3,4,5

    if not common_values_to_plot.empty:
        sns.barplot(x=common_values_to_plot.index, y=common_values_to_plot.values, hue=common_values_to_plot.index,
                    palette="viridis", dodge=False, legend=False)
        plt.title(f'Frequency of Common {target_variable} Values (0-5)')
        plt.xlabel(f'{target_variable} Value')
        plt.ylabel('Frequency (Count)')
        plt.xticks(rotation=0)  # Keep x-axis labels horizontal for small numbers
        plt.grid(axis='y', linestyle='--')
        # You can save this plot to include in your presentation
        # plt.savefig('canceledqty_common_value_counts.png', bbox_inches='tight')
        plt.show()
    else:
        print(f"No common values (0-5) found or value_counts_data is empty for {target_variable}.")

    # Plot 2: Histogram for an overall view, especially the tail
    # For data heavily skewed at 0, a log scale on the y-axis can help visualize less frequent values.
    # However, be cautious as log scale can sometimes be misleading if not explained.
    plt.figure(figsize=(12, 7))
    # Determine appropriate bins. If max is large, many bins might be needed, or group larger values.
    # For CanceledQty ranging up to 195, but mostly 0s and 1s:
    max_val = df[target_variable].max()
    if pd.isna(max_val):  # Handle case where max_val might be NaN if column is all NaN
        max_val = 0

    # Create bins that are more granular for small values and wider for larger values
    if max_val > 0:
        bins = list(range(0, min(int(max_val) + 2, 11)))  # Bins 0, 1, ..., 10
        if max_val > 10:
            bins.extend(list(range(10, min(int(max_val) + 2, 51), 5)))  # Bins 10, 15, ..., 50
        if max_val > 50:
            bins.extend(list(range(50, int(max_val) + 2, 25)))  # Bins 50, 75, ..., up to max
        # Ensure bins are unique and sorted, and include the max value if it's not covered
        bins = sorted(list(set(bins)))
        if bins[-1] < max_val:
            bins.append(max_val + 1)  # make sure the last bin includes the max value
    else:
        bins = [0, 1]  # Default if max_val is 0 or NaN

    sns.histplot(df[target_variable].dropna(), bins=bins, kde=False, color="skyblue")
    plt.title(f'Overall Distribution of {target_variable}')
    plt.xlabel(target_variable)
    plt.ylabel('Frequency')
    # Optional: Use a log scale for the y-axis if counts vary dramatically
    # plt.yscale('log')
    # if plt.gca().get_yscale() == 'log':
    #     plt.ylabel('Frequency (Log Scale)')
    plt.grid(axis='y', linestyle='--')
    plt.grid(axis='x', linestyle=':')
    # You can save this plot
    # plt.savefig('canceledqty_histogram_overall.png', bbox_inches='tight')
    plt.show()

    print(f"\nReminder: The '{target_variable}' distribution is highly skewed.")
    print("The bar plot of common values clearly shows the dominance of low cancellation numbers.")
    print("The histogram gives a broader view of the tail.")

else:
    print(f"Column '{target_variable}' not found in DataFrame or is not numeric.")

