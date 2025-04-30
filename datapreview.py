import pandas as pd

# Replace 'meal_orders.csv' with the actual path to your file
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