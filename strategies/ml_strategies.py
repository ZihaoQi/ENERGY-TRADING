import pandas as pd
from typing import Any
from base_classes import BaseStrategy, Position
from config import positive_target_class, neutral_target_class, negative_target_class, price_col


class MLPredictionStrategy(BaseStrategy):
    """
    Machine Learning based trading strategy using predicted probabilities
    """
    def __init__(self, model, feature_cols: list[str], target_col: str,
                 holding_period: int, threshold: float, take_profit_pct: float, stop_loss_pct: float):
        """
        Initialize the ML prediction strategy
        
        Args:
            model: Trained ML model with predict_proba method
            feature_cols: List of feature column names
            target_col: Target column name (for price data)
            holding_period: Number of bars to hold positions
            threshold: Probability threshold for signal generation
        """
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.holding_period = holding_period
        self.threshold = threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
    
    def generate_signals(self, df: pd.DataFrame, idx: int) -> list[dict[str, Any]]:
        """
        Generate trading signals based on ML model predictions
        
        Args:
            df: DataFrame with market data
            idx: Current index in the dataframe
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Get features for current row
        X_current = df.iloc[idx:idx+1][self.feature_cols]
        
        # Get prediction probabilities
        proba = self.model.predict_proba(X_current)[0]
        
        # Get class mapping
        classes = self.model.model.classes_
        class_proba = dict(zip(classes, proba))
        
        # Check for long signal (positive class with high probability)
        if positive_target_class in class_proba and class_proba[positive_target_class] > self.threshold:
            signals.append({
                'direction': 'long',
                'metadata': {
                    'probability': class_proba[positive_target_class],
                    'predicted_class': positive_target_class
                }
            })
        
        # Check for short signal (negative class with high probability)
        elif negative_target_class in class_proba and class_proba[negative_target_class] > self.threshold:
            signals.append({
                'direction': 'short',
                'metadata': {
                    'probability': class_proba[negative_target_class],
                    'predicted_class': negative_target_class
                }
            })
        
        return signals
    
    def should_close_position(self, position: Position, df: pd.DataFrame, 
                             current_idx: int) -> bool:
        """
        Close positions after holding_period bars
        
        Args:
            position: The open position
            df: DataFrame with market data
            current_idx: Current index in the dataframe
            
        Returns:
            True if position should be closed
        """

        current_price = df.iloc[current_idx][price_col]
        
        if position.direction == 'long':
            pct_change = ((current_price - position.entry_price) / position.entry_price) * 100
        else:  # short
            pct_change = ((position.entry_price - current_price) / position.entry_price) * 100
        
        # Check stop loss
        if pct_change <= -self.stop_loss_pct:
            return True
        
        if pct_change >= self.take_profit_pct:
            return True
        
        if current_idx - position.entry_idx >= self.holding_period:
            return True 


class TimeBasedExitStrategy(BaseStrategy):
    """
    Simple time-based exit strategy (example of another strategy)
    """
    def __init__(self, signal_func, holding_period: int):
        """
        Args:
            signal_func: Function that takes (df, idx) and returns direction or None
            holding_period: Number of bars to hold positions
        """
        self.signal_func = signal_func
        self.holding_period = holding_period
    
    def generate_signals(self, df: pd.DataFrame, idx: int) -> list[dict[str, Any]]:
        """Generate signals using custom function"""
        direction = self.signal_func(df, idx)
        
        if direction in ['long', 'short']:
            return [{'direction': direction, 'metadata': {}}]
        return []
    
    def should_close_position(self, position: Position, df: pd.DataFrame, 
                             current_idx: int) -> bool:
        """Close after holding_period bars"""
        return (current_idx - position.entry_idx) >= self.holding_period


class StopLossStrategy(BaseStrategy):
    """
    Strategy with stop-loss and take-profit exits
    """
    def __init__(self, signal_func, stop_loss_pct: float = 2.0, 
                 take_profit_pct: float = 5.0, max_holding_period: int = 20):
        """
        Args:
            signal_func: Function that takes (df, idx) and returns direction or None
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_holding_period: Maximum bars to hold
        """
        self.signal_func = signal_func
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_period = max_holding_period
    
    def generate_signals(self, df: pd.DataFrame, idx: int) -> list[dict[str, Any]]:
        """Generate signals using custom function"""
        direction = self.signal_func(df, idx)
        
        if direction in ['long', 'short']:
            return [{'direction': direction, 'metadata': {}}]
        return []
    
    def should_close_position(self, position: Position, df: pd.DataFrame, 
                             current_idx: int) -> bool:
        """
        Close on stop-loss, take-profit, or max holding period
        """
        current_price = df.iloc[current_idx]['Close']
        
        if position.direction == 'long':
            pct_change = ((current_price - position.entry_price) / position.entry_price) * 100
        else:  # short
            pct_change = ((position.entry_price - current_price) / position.entry_price) * 100
        
        # Check stop loss
        if pct_change <= -self.stop_loss_pct:
            return True
        
        # Check take profit
        if pct_change >= self.take_profit_pct:
            return True
        
        # Check max holding period
        if (current_idx - position.entry_idx) >= self.max_holding_period:
            return True
        
        return False