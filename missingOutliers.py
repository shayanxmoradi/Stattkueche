import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load your actual CSV file ---
# !!! Replace 'YOUR_FILE_NAME.csv' with the actual path to your 1GB CSV file !!!
file_path = "venvx/annonymdata.csv"
print(f"Attempting to load data from: {file_path}")

try:
    # Load the CSV file into a pandas DataFrame
    # For large files, you might need to consider memory optimization techniques:
    # 1. Specify dtypes: If you know the data types of your columns, specifying them can save memory.
    #    Example: dtype={'SchoolID': 'category', 'OrderQty': 'int32', ...}
    # 2. Load in chunks: If the file is too large to fit in memory at once.
    #    Example: pd.read_csv(file_path, chunksize=100000) - you'd then process each chunk.
    # For now, we'll try a direct load.
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
    print(f"DataFrame shape: {df.shape}")
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info(verbose=True, show_counts=True) # Provides a good overview of columns, non-null counts, and dtypes

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path and name.")
    exit() # Exit the script if the file isn't found
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    exit() # Exit if there's another loading error


# --- 1. Missing Values ---
print("\n\n--- Missing Value Analysis ---")

# Define critical columns based on your dataset structure
# Adjust this list if your column names are slightly different
critical_cols_missing = ['CanceledQty', 'DateOfService', 'OrderQty', 'SchoolID', 'DateOfCancel', 'MenuPrice', 'MenuSubsidy']
# Filter the list to include only columns that actually exist in the loaded DataFrame
critical_cols_missing = [col for col in critical_cols_missing if col in df.columns]

if not critical_cols_missing:
    print("Warning: None of the defined critical columns for missing value analysis were found in the DataFrame.")
else:
    # Calculate the percentage of missing values for each critical column
    missing_percentages = df[critical_cols_missing].isnull().mean() * 100

    print("\nPercentage of missing values per critical column:")
    if not missing_percentages.empty:
        for col, percentage in missing_percentages.items():
            print(f"- {col}: {percentage:.2f}%")
            if col == 'DateOfCancel' and percentage > 0:
                # This is where you'd put the string for your presentation placeholder
                print(f"  (Placeholder for presentation: '{percentage:.2f}% of DateOfCancel entries were missing.')")
    else:
        print("No missing values found in the specified critical columns or critical columns list is empty.")

    # Example: How to get the raw count of missing DateOfCancel
    if 'DateOfCancel' in df.columns:
        missing_date_cancel_count = df['DateOfCancel'].isnull().sum()
        total_date_cancel_count = len(df['DateOfCancel'])
        print(f"\nRaw count of missing DateOfCancel: {missing_date_cancel_count} out of {total_date_cancel_count}")


print("\nStrategies for handling missing values (for discussion):")
print("- Imputation: Fill with mean/median/mode, a constant, or using advanced methods.")
print("- Removal: Delete rows or columns (use with caution).")

# --- 2. Data Type Checks and Potential Conversions ---
print("\n\n--- Data Type Checks & Potential Conversions ---")
print("Initial data types (from df.info() above or df.dtypes):")
print(df.dtypes)

# Convert date columns to datetime objects
# Adjust this list based on your actual date column names
date_cols_to_convert = ['DateOfService', 'DateOfOrder', 'DateOfCancel']
for col in date_cols_to_convert:
    if col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"Attempting to convert {col} to datetime...")
            df[col] = pd.to_datetime(df[col], errors='coerce') # 'coerce' will turn unparseable dates into NaT
            print(f"Successfully converted {col} to datetime. NaNs introduced by coercion: {df[col].isnull().sum()}")
        else:
            print(f"{col} is already a datetime type.")
    else:
        print(f"Warning: Date column '{col}' not found in DataFrame.")

# Convert 'MenuPrice' and 'MenuSubsidy' to numeric, handling potential currency symbols
currency_cols = ['MenuPrice', 'MenuSubsidy']
for col in currency_cols:
    if col in df.columns:
        if df[col].dtype == 'object': # Only attempt conversion if it's an object type
            print(f"Attempting to convert currency column '{col}' to numeric...")
            # Common currency symbols and thousands separators to remove
            # Adjust the regex if your currency format is different (e.g., using '.' as thousands separator)
            df[col] = df[col].astype(str).str.replace(r'[€\$£,]', '', regex=True)
            # Handle potential different decimal separators, e.g., German format ',' as decimal
            # If your numbers use ',' as decimal and '.' as thousands, you'd first remove '.' then replace ',' with '.'
            # Assuming standard '.' as decimal after removing currency symbols and commas:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Successfully converted {col} to numeric. NaNs introduced by coercion: {df[col].isnull().sum()}")
        elif pd.api.types.is_numeric_dtype(df[col]):
            print(f"Currency column '{col}' is already numeric.")
        else:
            print(f"Currency column '{col}' is of type {df[col].dtype}, not attempting string-based currency conversion.")
    else:
        print(f"Warning: Currency column '{col}' not found in DataFrame.")


print("\nData types after potential conversions:")
print(df.dtypes)


# --- 3. Outlier Detection (Example for 'OrderQty') ---
print("\n\n--- Outlier Detection Analysis (Example for 'OrderQty') ---")
column_to_check_outliers = 'OrderQty'

if column_to_check_outliers in df.columns:
    if pd.api.types.is_numeric_dtype(df[column_to_check_outliers]):
        print(f"\nAnalyzing outliers for '{column_to_check_outliers}':")
        Q1 = df[column_to_check_outliers].quantile(0.25)
        Q3 = df[column_to_check_outliers].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"Lower bound for outliers: {lower_bound:.2f}")
        print(f"Upper bound for outliers: {upper_bound:.2f}")

        outliers_iqr = df[(df[column_to_check_outliers] < lower_bound) | (df[column_to_check_outliers] > upper_bound)]
        print(f"Number of outliers detected by IQR method in {column_to_check_outliers}: {len(outliers_iqr)}")

        if not outliers_iqr.empty:
            print(f"Sample outliers from {column_to_check_outliers} (IQR):")
            print(outliers_iqr[[column_to_check_outliers]].head())
            # Example for presentation placeholder
            print(f"  (Placeholder for presentation: 'Outliers in {column_to_check_outliers} were detected (e.g., values like {outliers_iqr[column_to_check_outliers].iloc[0]})...')")

        # Check for obviously invalid values (e.g., negative quantities)
        invalid_values = df[df[column_to_check_outliers] < 0]
        if not invalid_values.empty:
            print(f"\nInvalid (e.g., negative) values found in {column_to_check_outliers}: {len(invalid_values)}")
            print(invalid_values[[column_to_check_outliers]].head())
            print(f"  (Placeholder for presentation: 'Additionally, {len(invalid_values)} records with invalid negative {column_to_check_outliers} were found.')")

        # Visual Method: Box Plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column_to_check_outliers])
        plt.title(f'Box Plot of {column_to_check_outliers} for Outlier Detection')
        plt.xlabel(column_to_check_outliers)
        plt.grid(True)
        # To save the figure: plt.savefig('boxplot_orderqty.png')
        plt.show()
        print("Visual inspection of the box plot helps identify outliers (points beyond whiskers).")

    else:
        print(f"Column '{column_to_check_outliers}' is not numeric. Cannot perform outlier detection without prior cleaning/conversion.")
else:
    print(f"Warning: Column '{column_to_check_outliers}' for outlier analysis not found in DataFrame.")


print("\nStrategies for handling outliers (for discussion):")
print("- Removal, Transformation (log, sqrt), Capping/Winsorizing, Binning.")

# --- 4. Univariate Distribution (Example for 'OrderQty') ---
print("\n\n--- Univariate Distribution (Example for 'OrderQty') ---")
column_to_plot_distribution = 'OrderQty'

if column_to_plot_distribution in df.columns and pd.api.types.is_numeric_dtype(df[column_to_plot_distribution]):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_to_plot_distribution].dropna(), kde=True, bins=50) # dropna() for robustness, increased bins
    plt.title(f'Distribution of {column_to_plot_distribution}')
    plt.xlabel(column_to_plot_distribution)
    plt.ylabel('Frequency')
    plt.grid(True)
    # To save the figure: plt.savefig('histogram_orderqty.png')
    plt.show()

    print(f"\nSummary statistics for {column_to_plot_distribution}:")
    print(df[column_to_plot_distribution].describe())
    # Example for presentation: "The distribution of OrderQty showed a mean of X, a median of Y, and was [e.g., right-skewed]."
else:
    print(f"Cannot plot distribution for '{column_to_plot_distribution}'. It's either not in DataFrame or not numeric.")

print("\n\n--- End of Data Quality Initial Exploration Script ---")

