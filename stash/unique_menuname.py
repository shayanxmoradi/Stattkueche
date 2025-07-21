import pandas as pd

# Assume your DataFrame is already loaded, e.g.:
file_path =  "/Users/shayan/Desktop/IDS2/Stattkueche/venvx/AnnonymData.csv"
df = pd.read_csv(file_path)

# Check if the 'MenuName' column exists
if 'MenuName' in df.columns:
    # Get unique values from the 'MenuName' column
    unique_menu_names = df['MenuName'].unique()

    # Print the number of unique menu names
    print(f"Number of unique menu names: {len(unique_menu_names)}")

    # Print all unique menu names
    print("\nUnique menu names:")
    for name in unique_menu_names:
        print(name)
else:
    print("Error: 'MenuName' column not found in the DataFrame.")