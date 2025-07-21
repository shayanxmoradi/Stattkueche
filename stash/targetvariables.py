import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load your actual CSV file ---
# !!! Replace 'YOUR_FILE_NAME.csv' with the actual path to your 1GB CSV file !!!
file_path = input_file = "venvx/annonymdata.csv"       # Your original 1GB CSV

print(f"Attempting to load data from: {file_path}")

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
    print(f"DataFrame shape: {df.shape}")

    # --- Apply initial data type conversions (essential for this script) ---
    print("\n--- Applying Essential Data Type Conversions ---")
    date_cols_to_convert = ['DateOfService', 'DateOfOrder', 'DateOfCancel']
    for col in date_cols_to_convert:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
    print("Date columns converted (if not already datetime).")

    currency_cols = ['MenuPrice', 'MenuSubsidy']
    for col in currency_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(r'[€\$£,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Currency columns converted to numeric (if they were objects).")
    # Assuming OrderQty and CanceledQty are already int64 as per your previous output
    # If not, ensure they are numeric:
    # df['OrderQty'] = pd.to_numeric(df['OrderQty'], errors='coerce')
    # df['CanceledQty'] = pd.to_numeric(df['CanceledQty'], errors='coerce')


except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path and name.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- 1. Target Variable Analysis: CanceledQty ---
print("\n\n--- Target Variable Analysis: CanceledQty ---")
target_variable = 'CanceledQty'

if target_variable in df.columns and pd.api.types.is_numeric_dtype(df[target_variable]):
    print(f"\nSummary statistics for '{target_variable}':")
    print(df[target_variable].describe())

    # For the placeholder: [Placeholder: Target distribution...]
    # Describe the distribution based on these stats.
    # Example: "The CanceledQty ranges from X to Y, with a mean of Z. The median is W,
    #           indicating [skewness if mean and median are different]."

    # Visualization of CanceledQty distribution
    # Since CanceledQty is likely discrete and might be skewed (many 0s),
    # a bar plot of its value counts might be more informative than a histogram for common values.
    plt.figure(figsize=(12, 7))

    # Option 1: Bar plot for value counts (good for discrete, low-cardinality numbers)
    # Let's see the counts for the most common cancellation quantities
    top_n_canceled_qty = df[target_variable].value_counts().nlargest(10) # Top 10 cancellation quantities
    if not top_n_canceled_qty.empty:
        sns.barplot(x=top_n_canceled_qty.index, y=top_n_canceled_qty.values, palette="viridis")
        plt.title(f'Top 10 Value Counts for {target_variable}')
        plt.xlabel(target_variable + ' Value')
        plt.ylabel('Frequency (Count)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')
        # plt.savefig('canceledqty_value_counts.png') # To save the figure
        plt.show()
    else:
        print(f"No data to plot for {target_variable} value counts.")


    # Option 2: Histogram (if there's a wider range of values or you want to see bins)
    # Be mindful that if most values are 0 or 1, a histogram might not be as clear as value counts.
    # Filter out potential extreme values for a more focused histogram if necessary.
    # For instance, if 99% of CanceledQty is < 5, but max is 100.
    # lower_quantile = df[target_variable].quantile(0.01)
    # upper_quantile = df[target_variable].quantile(0.99)
    # filtered_canceled_qty = df[(df[target_variable] >= lower_quantile) & (df[target_variable] <= upper_quantile)][target_variable]

    plt.figure(figsize=(12, 7))
    # Use a sensible number of bins. If max CanceledQty is small, use that for bins.
    # max_val = df[target_variable].max()
    # bins = range(0, int(max_val) + 2) if max_val < 50 else 50 # Example binning
    sns.histplot(df[target_variable].dropna(), kde=False, discrete=True) # discrete=True for integer values
    plt.title(f'Distribution of {target_variable}')
    plt.xlabel(target_variable)
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.savefig('canceledqty_histogram.png') # To save the figure
    plt.show()
    print(f"Note: The CanceledQty distribution is likely heavily skewed, with many orders having 0 cancellations.")
    print(f"      The value_counts plot shows the frequency of specific cancellation numbers.")
    print(f"      The histogram gives a binned overview.")

else:
    print(f"Target variable '{target_variable}' not found or not numeric in the DataFrame.")


# --- 2. Distributions of Other Key Features ---
print("\n\n--- Distributions of Other Key Features ---")

# a) OrderQty (We already have this from previous script, but for completeness)
feature_orderqty = 'OrderQty'
if feature_orderqty in df.columns and pd.api.types.is_numeric_dtype(df[feature_orderqty]):
    print(f"\nSummary statistics for '{feature_orderqty}' (already analyzed):")
    print(df[feature_orderqty].describe())
    # Placeholder: [Placeholder: e.g., OrderQty dist...]
    # "The OrderQty is heavily concentrated at 1 (median=1, 75th percentile=1),
    #  ranging from 0 to 120, with a mean of ~0.88."
    # (Histogram was generated in the previous script)
else:
    print(f"Feature '{feature_orderqty}' not found or not numeric.")


# b) MenuPrice Distribution
feature_menuprice = 'MenuPrice'
if feature_menuprice in df.columns and pd.api.types.is_numeric_dtype(df[feature_menuprice]):
    print(f"\nSummary statistics for '{feature_menuprice}':")
    print(df[feature_menuprice].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature_menuprice].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {feature_menuprice}')
    plt.xlabel(feature_menuprice)
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.savefig('menuprice_histogram.png') # To save the figure
    plt.show()
    # Describe its distribution for the presentation.
else:
    print(f"Feature '{feature_menuprice}' not found or not numeric.")

# c) Number of Orders per SchoolID (Top N)
feature_schoolid = 'SchoolID'
if feature_schoolid in df.columns:
    print(f"\nAnalyzing '{feature_schoolid}':")
    school_order_counts = df[feature_schoolid].value_counts()
    print(f"Number of unique SchoolIDs: {school_order_counts.nunique()}")
    print("\nTop 10 SchoolIDs by number of orders:")
    print(school_order_counts.head(10))

    plt.figure(figsize=(12, 7))
    top_n_schools = school_order_counts.nlargest(15) # Top 15 schools
    if not top_n_schools.empty:
        sns.barplot(x=top_n_schools.index, y=top_n_schools.values, palette="mako")
        plt.title(f'Top 15 SchoolIDs by Order Volume')
        plt.xlabel(feature_schoolid)
        plt.ylabel('Number of Orders')
        plt.xticks(rotation=75) # Rotate labels for better readability if SchoolIDs are long
        plt.grid(axis='y', linestyle='--')
        # plt.savefig('schoolid_order_counts.png') # To save the figure
        plt.show()
    else:
        print(f"No data to plot for {feature_schoolid} counts.")
    # Describe this for the presentation: "Analysis of SchoolID shows X unique schools,
    # with the top schools accounting for a significant portion of orders."
else:
    print(f"Feature '{feature_schoolid}' not found.")


print("\n\n--- End of Script ---")
