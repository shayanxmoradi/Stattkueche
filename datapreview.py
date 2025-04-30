import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error #
pd.set_option('display.max_columns', None)
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
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



import pandas as pd

# Assuming 'daily_cancellations' is your DataFrame from the previous step

print("Starting Feature Engineering...")

# Create time-based features from DateOfService
daily_cancellations['day_of_week'] = daily_cancellations['DateOfService'].dt.dayofweek # Monday=0, Sunday=6
daily_cancellations['month'] = daily_cancellations['DateOfService'].dt.month
daily_cancellations['year'] = daily_cancellations['DateOfService'].dt.year
daily_cancellations['day_of_year'] = daily_cancellations['DateOfService'].dt.dayofyear
daily_cancellations['day_of_month'] = daily_cancellations['DateOfService'].dt.day
# Add a simple numerical representation of the date (e.g., Unix timestamp or ordinal)
# This helps capture linear trends over time if they exist
daily_cancellations['date_ordinal'] = daily_cancellations['DateOfService'].apply(lambda x: x.toordinal())


print("\nDataFrame with time-based features:")
print(daily_cancellations.head())


# Handle SchoolID using one-hot encoding
# drop_first=True is used to avoid multicollinearity
school_dummies = pd.get_dummies(daily_cancellations['SchoolID'], prefix='SchoolID', drop_first=True)

# Concatenate the new dummy columns with the main DataFrame
daily_cancellations = pd.concat([daily_cancellations, school_dummies], axis=1)

# Drop the original 'SchoolID' column as we now have the dummy columns
daily_cancellations.drop('SchoolID', axis=1, inplace=True)

print("\nDataFrame after one-hot encoding SchoolID:")
print(daily_cancellations.head())
print("\nInfo after feature engineering:")
daily_cancellations.info()

print("\nFeature Engineering Complete.")






# Assuming 'daily_cancellations' is your DataFrame after feature engineering

print("\nStarting Data Splitting (Chronological)...")

# Ensure data is sorted by date to guarantee a correct chronological split
daily_cancellations = daily_cancellations.sort_values('DateOfService')

# Define features (X) and target (y)
# We drop 'DateOfService' as it's no longer needed directly in the model features
X = daily_cancellations.drop(['DateOfService', 'NumCancellations'], axis=1)
y = daily_cancellations['NumCancellations']

# Determine the split point based on the number of rows.
# A common split is 80% for training and 20% for testing.
split_index = int(len(daily_cancellations) * 0.8)

# Split the data chronologically using iloc
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Original dataset shape: {daily_cancellations.shape}")
print(f"Training set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")
print(f"Training target shape (y_train): {y_train.shape}")
print(f"Testing target shape (y_test): {y_test.shape}")

print("\nData Splitting Complete.")




# Assuming X_train, X_test, y_train, y_test are your DataFrames/Series from Step 4

print("\nStarting Model Training (Linear Regression)...")

# Create a Linear Regression model instance
model = LinearRegression()

# Train the model using the training data
# The model learns the coefficients (weights) for each feature
model.fit(X_train, y_train)

print("Model Training Complete.")

# --- Make Predictions ---
# Now that the model is trained, let's make predictions on the test data
print("Making Predictions on Test Data...")
predictions = model.predict(X_test)
print("Predictions Complete.")

# You can look at the first few predictions
print("\nFirst 10 predictions:")
print(predictions[:10])

# And compare them to the actual values in the test set
print("\nFirst 10 actual values (y_test):")
print(y_test.head(10).tolist()) # Convert to list for easier comparison






# Assuming predictions and y_test are available from Step 5

print("\nStarting Model Evaluation...")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}") # .2f formats to 2 decimal places

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print("\nModel Evaluation Complete.")






