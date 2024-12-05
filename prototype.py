import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from neuralprophet import NeuralProphet
import holidays

def apply_kalman_filter(y):
    """Applies a Kalman filter for trend estimation on the target series."""
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, n_dim_state=1)
    kf = kf.em(y, n_iter=10)
    state_means, _ = kf.filter(y)
    return state_means.flatten()

def get_indian_holidays(start_date, end_date):
    """Get Indian holidays between start and end date."""
    in_holidays = holidays.India()
    holiday_dates = []
    holiday_names = []
    
    date_range = pd.date_range(start=start_date, end=end_date)
    for date in date_range:
        if date in in_holidays:
            holiday_dates.append(date)
            holiday_names.append(in_holidays[date])
    
    return pd.DataFrame({
        'ds': holiday_dates,
        'holiday': holiday_names,
        'lower_window': -1,
        'upper_window': 1
    })

def calculate_rolling_features(series, windows=[7, 30]):
    """Calculate rolling statistics for multiple windows."""
    features = {}
    for window in windows:
        features[f'roll_mean_{window}'] = series.rolling(window=window, min_periods=1).mean()
        features[f'roll_std_{window}'] = series.rolling(window=window, min_periods=1).std()
        features[f'roll_min_{window}'] = series.rolling(window=window, min_periods=1).min()
        features[f'roll_max_{window}'] = series.rolling(window=window, min_periods=1).max()
    return pd.DataFrame(features)

def add_seasonal_features(df):
    """Add seasonal features to the dataframe."""
    df['month_sin'] = np.sin(2 * np.pi * df['ds'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['ds'].dt.month / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['ds'].dt.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['ds'].dt.dayofweek / 7)
    df['quarter_sin'] = np.sin(2 * np.pi * df['ds'].dt.quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['ds'].dt.quarter / 4)
    return df

def validate_and_scale_data(df, regressor_columns):
    """Validate and scale the data to prevent numerical issues."""
    scaled_df = df.copy()
    
    # Handle missing values
    print(f"Missing values before cleaning:\n{scaled_df.isnull().sum()}")
    
    # Forward fill missing values in target variable
    scaled_df['y'] = scaled_df['y'].fillna(method='ffill')
    
    # Forward fill missing values in regressors
    for col in regressor_columns:
        scaled_df[col] = scaled_df[col].fillna(method='ffill')
        # Backward fill any remaining missing values at the start
        scaled_df[col] = scaled_df[col].fillna(method='bfill')
    
    print(f"Missing values after cleaning:\n{scaled_df.isnull().sum()}")
    
    # Scale the target variable
    y_min = scaled_df['y'].min()
    y_max = scaled_df['y'].max()
    scaled_df['y'] = (scaled_df['y'] - y_min) / (y_max - y_min)
    
    # Scale each regressor
    scaler_dict = {}
    for col in regressor_columns:
        col_min = scaled_df[col].min()
        col_max = scaled_df[col].max()
        if col_max - col_min != 0:
            scaled_df[col] = (scaled_df[col] - col_min) / (col_max - col_min)
            scaler_dict[col] = {'min': col_min, 'max': col_max}
        else:
            scaled_df[col] = scaled_df[col] - col_min
            scaler_dict[col] = {'min': col_min, 'max': col_min + 1}
    
    return scaled_df, scaler_dict, {'min': y_min, 'max': y_max}

def prepare_future_df(train_df, periods, regressor_columns, scaler_dict):
    """Prepare future dataframe with dynamic rolling features."""
    last_date = train_df['ds'].max()
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date,
        periods=periods + 1,
        freq='B'
    )[1:]
    
    # Create future dataframe
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add seasonal features
    future_df = add_seasonal_features(future_df)
    
    # Initialize target variable with last known value
    last_known_y = train_df['y'].iloc[-1]
    future_df['y'] = last_known_y
    
    # Get the last known values for all features
    last_known_values = {}
    for col in regressor_columns:
        if col not in ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'quarter_sin', 'quarter_cos']:
            last_known_values[col] = train_df[col].iloc[-1]
    
    # Initialize rolling features with last known values
    for col in regressor_columns:
        if col in last_known_values:
            future_df[col] = last_known_values[col]
    
    # Ensure no missing values in any column
    for col in future_df.columns:
        if future_df[col].isnull().any():
            # Forward fill any missing values
            future_df[col] = future_df[col].fillna(method='ffill')
            # If still any missing values (at the start), backward fill
            future_df[col] = future_df[col].fillna(method='bfill')
    
    # Print debugging information
    print("\nFuture DataFrame Check:")
    print(f"Shape: {future_df.shape}")
    print("Missing values:")
    print(future_df.isnull().sum())
    print("\nColumns present:", future_df.columns.tolist())
    
    return future_df

def create_model():
    """Create and configure the NeuralProphet model with optimized parameters."""
    model = NeuralProphet(
        n_forecasts=1,
        n_lags=28,  
        ar_layers=[3,3,3],
        optimizer="AdamW",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        learning_rate=11e-5,
        batch_size=64,
        epochs=2001,
        loss_func='Huber',
        seasonality_mode='multiplicative',
        seasonality_reg=0.1,
        growth='linear',
        n_changepoints=120,
        changepoints_range=0.9,
        trend_reg=0.1,
        impute_missing=True,  
        drop_missing=False
    )
    
    # Add Indian holidays
    model.add_country_holidays(country_name='IN', mode="multiplicative")
    
    return model

def plot_forecast(train_df, forecast, holidays_df, y_scaler):
    """Create detailed visualization of the forecast with properly scaled historical data."""
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Unscale historical data
    historical_unscaled = train_df['y'] * (y_scaler['max'] - y_scaler['min']) + y_scaler['min']
    
    # Plot full history and forecast
    ax1.plot(train_df['ds'], historical_unscaled, label='Historical', alpha=0.7)
    ax1.plot(forecast['ds'], forecast['yhat1_unscaled'], label='Forecast', alpha=0.7)
    
    # Plot holidays
    for _, holiday in holidays_df.iterrows():
        ax1.axvline(x=holiday['ds'], color='r', alpha=0.2, linestyle='--')
    
    ax1.set_title('Complete Price Forecast with Kalman Filter Trend and Indian Holidays')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot recent history and forecast
    recent_data = train_df[train_df['ds'] >= train_df['ds'].max() - pd.Timedelta(days=90)]
    recent_unscaled = recent_data['y'] * (y_scaler['max'] - y_scaler['min']) + y_scaler['min']
    
    ax2.plot(recent_data['ds'], recent_unscaled, label='Recent History', alpha=0.7)
    ax2.plot(forecast['ds'], forecast['yhat1_unscaled'], label='Forecast', alpha=0.7)
    
    # Add confidence intervals if available
    if 'yhat1_lower' in forecast.columns and 'yhat1_upper' in forecast.columns:
        lower = forecast['yhat1_lower'] * (y_scaler['max'] - y_scaler['min']) + y_scaler['min']
        upper = forecast['yhat1_upper'] * (y_scaler['max'] - y_scaler['min']) + y_scaler['min']
        ax2.fill_between(forecast['ds'], lower, upper, alpha=0.2, label='95% Confidence Interval')
    
    ax2.set_title('Recent History and Forecast (Last 60 Days)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 

def main():
    try:
        # Load data
        df = pd.read_csv('prototype_Final.csv')
        df = df.rename(columns={'Date': 'ds', 'Price': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        
        # Check for and handle gaps in dates
        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='B')
        df = df.set_index('ds').reindex(date_range).reset_index()
        df = df.rename(columns={'index': 'ds'})
        
        # Apply Kalman filter
        df['price_trend_kalman'] = apply_kalman_filter(df['y'].fillna(method='ffill'))
        
        # Add seasonal features
        df = add_seasonal_features(df)
        
        # Add rolling features
        rolling_features = calculate_rolling_features(df['y'].fillna(method='ffill'))
        df = pd.concat([df, rolling_features], axis=1)
        
        # Define regressor columns
        regressor_columns = [
            'roll_mean_7', 'roll_std_7', 'roll_min_7', 'roll_max_7',
            'roll_mean_30', 'roll_std_30', 'roll_min_30', 'roll_max_30',
            'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
            'quarter_sin', 'quarter_cos', 'price_trend_kalman'
        ]
        
        # Prepare and scale training data
        train_df = df[['ds', 'y'] + regressor_columns].copy()
        train_df, scaler_dict, y_scaler = validate_and_scale_data(train_df, regressor_columns)
        
        # Print data quality information
        print("\nData Quality Check:")
        print(f"Total rows: {len(train_df)}")
        print(f"Date range: {train_df['ds'].min()} to {train_df['ds'].max()}")
        print(f"Missing values:\n{train_df.isnull().sum()}")
        
        # Get Indian holidays
        holidays_df = get_indian_holidays(
            train_df['ds'].min(),
            train_df['ds'].max() + pd.Timedelta(days=90)
        )
        
        # Create and fit model
        model = create_model()
        
        # Add regressors
        for col in regressor_columns:
            model.add_future_regressor(col, normalize=True)
        
        # Fit the model using train_df instead of scaled_train_df
        metrics = model.fit(train_df, freq='B')
        
        # Prepare future dataframe and make predictions
        periods = 60
        future_df = prepare_future_df(train_df, periods, regressor_columns, scaler_dict)
        
        # Validate future_df before prediction
        print("\nValidating future dataframe...")
        missing_cols = set(regressor_columns) - set(future_df.columns)
        if missing_cols:
            print(f"Warning: Missing columns in future_df: {missing_cols}")
            raise ValueError(f"Future dataframe is missing required columns: {missing_cols}")
        
        # Check for any missing values
        if future_df.isnull().any().any():
            print("Warning: Found missing values in future_df")
            print(future_df.isnull().sum())
            # Fill any remaining missing values
            future_df = future_df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure all required columns are present and have no missing values
        required_cols = ['ds', 'y'] + regressor_columns
        for col in required_cols:
            if col not in future_df.columns:
                raise ValueError(f"Missing required column: {col}")
            if future_df[col].isnull().any():
                raise ValueError(f"Column {col} contains missing values")
        
        forecast = model.predict(future_df)
        
        # Unscale predictions
        forecast['yhat1_unscaled'] = forecast['yhat1'] * (y_scaler['max'] - y_scaler['min']) + y_scaler['min']
        
        # Print validation metrics
        print("\nPrediction validation:")
        print("Total predictions:", len(forecast))
        print("NaN values in predictions:", forecast['yhat1'].isnull().sum())
        print("\nFirst few predictions:")
        print(forecast[['ds', 'yhat1_unscaled']].head())
        
        # Create visualization and display it
        plot_forecast(train_df, forecast, holidays_df, y_scaler)
        
        # Save predictions and metrics (keeping these file outputs)
        forecast[['ds', 'yhat1_unscaled']].to_csv('price_forecast_results.csv', index=False)
        pd.DataFrame(metrics).to_csv('model_metrics.csv')
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        print("\nDebug information:")
        print(f"Train df shape: {train_df.shape}")
        print(f"Train df columns: {train_df.columns.tolist()}")
        print(f"Future df shape: {future_df.shape if 'future_df' in locals() else 'Not created'}")
        print(f"Future df columns: {future_df.columns.tolist() if 'future_df' in locals() else 'Not created'}")
        raise e
        

if __name__ == "__main__":
    main()
    
