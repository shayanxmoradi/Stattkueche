import pandas as pd

try:
    addres = 'venvx/AnnonymData.csv'

    df = pd.read_csv(addres)
    print("CSV file loaded successfully!")
    print("First 5 rows of the data:")
    print(df.head())
    print("\nColumn information:")
    df.info()
except FileNotFoundError:
    print("Error: The file 'meal_orders.csv' was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")


# address = 'venvx/AnnonymData.csv'
#
# df = pd.read_csv(address)



try:
    df['DateOfService'] = pd.to_datetime(df['DateOfService'])
    print("\n'DateOfService' column converted to datetime.")
except KeyError:
    print("\nError: 'DateOfService' column not found. Please check the exact column name for the service date.")
    # You might need to inspect df.columns to find the correct date column name
    print("Available columns:", df.columns.tolist())
except Exception as e:
    print(f"\nAn error occurred during date conversion: {e}")


# 2. Define the target variable: Number of cancellations per school per day
# We'll group by DateOfService and SchoolID and sum the CanceledQty

# Let's first check if 'CanceledQty' and 'SchoolID' columns exist
if 'CanceledQty' in df.columns and 'SchoolID' in df.columns:
    try:
        # Group by date and school and sum the canceled quantity
        daily_cancellations = df.groupby(['DateOfService', 'SchoolID'])['CanceledQty'].sum().reset_index()

        # Rename the sum column to something more descriptive
        daily_cancellations.rename(columns={'CanceledQty': 'NumCancellations'}, inplace=True)

        print("\nAggregated daily cancellations per school:")
        print(daily_cancellations.head(20))
        print("\nInfo on the aggregated data:")
        daily_cancellations.info()

    except Exception as e:
        print(f"\nAn error occurred during aggregation: {e}")
else:
    print("\nError: 'CanceledQty' or 'SchoolID' column not found. Please check the exact column names.")
    print("Available columns:", df.columns.tolist())




