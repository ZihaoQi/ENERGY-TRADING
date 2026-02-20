# FEATURE ENGINEERING

import numpy as np
from config import positive_target_class, neutral_target_class, negative_target_class

def create_features(df, target_col: str, lookback_period=1, return_threshold=0.005):
    """
    Create features and target variable(s)
    """
    df = df.copy()
    
    # Create lagged features
    df[target_col + '_lag1'] = df[target_col].shift(1)
    df[target_col + '_lag2'] = df[target_col].shift(2)
    df[target_col + '_lag5'] = df[target_col].shift(5)

    # Monthly realized vol (annualized)
    df[target_col + '_rv'] = np.log(df[target_col]).diff().rolling(21).std()

    # Calculate 5-day forward return
    df[target_col + '_future'] = df[target_col].shift(-lookback_period)
    df[target_col + '_future_return'] = ((df[target_col + '_future'] - df[target_col]) / df[target_col])
    
    # Create target variable based on return thresholds
    df['target'] = neutral_target_class  # Default: neutral
    df.loc[df[target_col + '_future_return'] > return_threshold, 'target'] = positive_target_class   # Positive
    df.loc[df[target_col + '_future_return'] < -return_threshold, 'target'] = negative_target_class  # Negative
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df