import pandas as pd
import numpy as np
import lightgbm as lgb
import sklearn

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
print(sklearn.__version__)

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error, handling cases where y_true is 0.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero for y_true
    # Replace 0s in y_true with a very small number or handle as per specific business logic
    # For simplicity here, we'll ignore entries where y_true is 0 for MAPE calculation
    mask = y_true != 0
    if not np.any(mask): # All true values are zero
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    print("Starting LightGBM forecasting script...")

    # --- 1. Load Data ---

    file_path =  'venvx/AnnonymData.csv'
    try:
        df_original = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Data loaded successfully.")
    df_original.info(verbose=True, show_counts=True)

    # --- 2. Core Data Preprocessing ---
    print("\nStarting core data preprocessing...")
    # Convert date columns
    date_cols = ['DateOfService', 'DateOfOrder', 'DateOfCancel']
    for col in date_cols:
        df_original[col] = pd.to_datetime(df_original[col], errors='coerce')

    # --- Monetary Column Cleaning Placeholder ---
    # If 'MenuPrice' or 'MenuSubsidy' are objects and need cleaning (e.g., removing '$', '€'):
    # Example:
    # df_original['MenuPrice'] = df_original['MenuPrice'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    # df_original['MenuSubsidy'] = df_original['MenuSubsidy'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    # For this script, we assume they are numeric or will be handled if aggregated later.

    # Create aggregated target variable: TotalCanceledQty per DateOfService
    df_agg = df_original.groupby('DateOfService')['CanceledQty'].sum().reset_index()  # Correct
    # df_agg = df_original.groupby('DateOfService')['CanceledQty'].sum().reset_name()
    df_agg = df_agg.rename(columns={'CanceledQty': 'TotalCanceledQty'})
    df_agg = df_agg.sort_values('DateOfService')

    # Ensure DateOfService is the index and we have a complete time series (e.g., daily)
    # If you know your data is continuous daily, you might not need asfreq rigorously.
    # But if there are gaps you want to fill (e.g., with 0s):
    df_agg = df_agg.set_index('DateOfService')
    df_agg = df_agg.asfreq('D', fill_value=0) # Fills missing days with 0 cancellations
    df_agg = df_agg.reset_index() # Bring DateOfService back as a column for feature engineering

    target_col = 'TotalCanceledQty'
    print(f"Aggregated data created. Target column: '{target_col}'.")

    # --- 3. Feature Engineering ---
    print("\nStarting feature engineering...")
    df_feat = df_agg.copy()
    df_feat['DateOfService'] = pd.to_datetime(df_feat['DateOfService']) # Ensure it's datetime

    # Date & Time Features
    df_feat['day_of_week'] = df_feat['DateOfService'].dt.dayofweek
    df_feat['day_of_month'] = df_feat['DateOfService'].dt.day
    df_feat['day_of_year'] = df_feat['DateOfService'].dt.dayofyear
    df_feat['week_of_year'] = df_feat['DateOfService'].dt.isocalendar().week.astype(int)
    df_feat['month'] = df_feat['DateOfService'].dt.month
    df_feat['year'] = df_feat['DateOfService'].dt.year
    df_feat['is_weekend'] = (df_feat['DateOfService'].dt.dayofweek >= 5).astype(int)
    # Consider these as categorical for LightGBM
    for col in ['day_of_week', 'month', 'year', 'is_weekend', 'week_of_year', 'day_of_month', 'day_of_year']:
         if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype('category')


    # Lag Features
    lag_periods = [1, 2, 3, 7, 14, 21, 28, 35, 42, 365] # Added a yearly lag
    for lag in lag_periods:
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)

    # Rolling Window Features
    window_sizes = [7, 14, 28, 365] # Added a yearly window
    for window in window_sizes:
        df_feat[f'{target_col}_roll_mean_{window}'] = df_feat[target_col].shift(1).rolling(window=window, min_periods=1).mean() # min_periods=1 handles initial NaNs better
        df_feat[f'{target_col}_roll_std_{window}'] = df_feat[target_col].shift(1).rolling(window=window, min_periods=1).std()
        df_feat[f'{target_col}_roll_min_{window}'] = df_feat[target_col].shift(1).rolling(window=window, min_periods=1).min()
        df_feat[f'{target_col}_roll_max_{window}'] = df_feat[target_col].shift(1).rolling(window=window, min_periods=1).max()

    # --- Placeholder for Aggregated Exogenous Features ---
    # This is where you would merge other aggregated features from your original dataset
    # E.g., df_feat = pd.merge(df_feat, daily_avg_menu_price_df, on='DateOfService', how='left')
    # E.g., df_feat = pd.merge(df_feat, daily_order_counts_df, on='DateOfService', how='left')
    # Ensure these new features are also handled for missing values.

    # Drop rows with NaNs created by lags/rolling windows (at the beginning of the series)
    initial_rows = len(df_feat)
    df_feat = df_feat.dropna()
    print(f"Dropped {initial_rows - len(df_feat)} rows due to NaNs from feature engineering.")

    if df_feat.empty:
        print("Error: DataFrame is empty after feature engineering and dropping NaNs. Please check your data and feature creation steps.")
        return

    # Set DateOfService as index for easy splitting and plotting
    df_feat = df_feat.set_index('DateOfService')
    print("Feature engineering complete.")

    # --- 4. Data Splitting (Time-Series Aware) ---
    print("\nSplitting data...")
    y = df_feat[target_col]
    X = df_feat.drop(columns=[target_col])

    # Ensure chronological order for splitting (already done by sorting and DateOfService index)
    # Define split point (e.g., last 10% for test, previous 10% for validation)
    # Adjust these fractions or use specific dates as needed
    test_fraction = 0.15
    val_fraction = 0.15

    num_samples = len(X)
    num_test_samples = int(num_samples * test_fraction)
    num_val_samples = int(num_samples * val_fraction)
    num_train_samples = num_samples - num_test_samples - num_val_samples

    if num_train_samples <= 0 or num_val_samples <=0 or num_test_samples <= 0:
        print("Error: Not enough data for train/validation/test split. Please check data size after preprocessing.")
        return

    X_train, y_train = X.iloc[:num_train_samples], y.iloc[:num_train_samples]
    X_val, y_val = X.iloc[num_train_samples:num_train_samples + num_val_samples], y.iloc[num_train_samples:num_train_samples + num_val_samples]
    X_test, y_test = X.iloc[num_train_samples + num_val_samples:], y.iloc[num_train_samples + num_val_samples:]

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

    # Identify categorical features for LightGBM
    # LightGBM can infer, but explicit passing is better.
    categorical_features_names = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
    print(f"Identified categorical features: {categorical_features_names}")


    # --- 5. LightGBM Model Training ---
    print("\nTraining LightGBM model...")
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features_names, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features_names, free_raw_data=False)

    # Initial parameters (these should be tuned later)
    params = {
        'objective': 'regression_l1',  # MAE, robust to outliers
        'metric': ['l1', 'rmse'],      # Evaluate on MAE (l1) and RMSE (l2/rmse)
        'boosting_type': 'gbdt',
        'n_estimators': 2000,          # Number of boosting rounds (trees)
        'learning_rate': 0.02,
        'num_leaves': 31,              # Max number of leaves in one tree
        'max_depth': -1,               # No limit on tree depth
        'min_child_samples': 20,       # Min number of data in one leaf
        'subsample': 0.8,              # Fraction of data to be used for fitting the individual base learners
        'colsample_bytree': 0.8,       # Fraction of features to be used for fitting the individual base learners
        'random_state': 42,
        'n_jobs': -1,                  # Use all available cores
        'verbose': -1,                 # Suppress training verbosity in basic LightGBM
        'feature_pre_filter': False    # Recommended by LightGBM docs for performance with many features
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True), # Stop if val score doesn't improve for 100 rounds
            lgb.log_evaluation(period=100) # Log metrics every 100 rounds
        ]
    )

    # --- 6. Prediction and Evaluation ---
    print("\nEvaluating model...")
    best_iteration = model.best_iteration if model.best_iteration else params['n_estimators']
    print(f"Best iteration found: {best_iteration}")

    y_pred_val = model.predict(X_val, num_iteration=best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=best_iteration)

    # Clip predictions to be non-negative as CanceledQty cannot be negative
    y_pred_val = np.maximum(0, y_pred_val)
    y_pred_test = np.maximum(0, y_pred_test)


    print("\n--- Validation Set Metrics ---")
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
    val_mape = calculate_mape(y_val, y_pred_val)
    print(f"Validation MAE (Mean Absolute Error): {val_mae:.4f}")
    print(f"Validation RMSE (Root Mean Squared Error): {val_rmse:.4f}")
    print(f"Validation MAPE (Mean Absolute Percentage Error): {val_mape:.2f}%")

    print("\n--- Test Set Metrics (Success Indication) ---")
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    test_mape = calculate_mape(y_test, y_pred_test)
    print(f"Test MAE (Mean Absolute Error): {test_mae:.4f}")
    print(f"Test RMSE (Root Mean Squared Error): {test_rmse:.4f}")
    print(f"Test MAPE (Mean Absolute Percentage Error): {test_mape:.2f}%")
    print("Lower MAE, RMSE, and MAPE indicate better performance.")

    # --- 7. Feature Importance ---
    print("\nPlotting feature importance...")
    lgb.plot_importance(model, figsize=(12, 10), max_num_features=30, importance_type='gain')
    plt.title("LightGBM Feature Importance (Gain)")
    plt.tight_layout()
    plt.show()

    # --- 8. Plot Predictions vs Actuals ---
    print("\nPlotting actuals vs predictions for the test set...")
    plt.figure(figsize=(18, 8))
    plt.plot(y_test.index, y_test, label='Actual Canceled Qty', marker='.', linestyle='-')
    plt.plot(y_test.index, y_pred_test, label='Predicted Canceled Qty (LightGBM)', marker='.', linestyle='--')
    plt.title(f'Test Set: Actual vs. Predicted Cancellations (MAE: {test_mae:.2f})')
    plt.xlabel('DateOfService')
    plt.ylabel('TotalCanceledQty')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nScript finished.")

if __name__ == '__main__':
    main()