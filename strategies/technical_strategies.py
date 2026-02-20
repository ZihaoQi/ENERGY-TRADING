from typing import Optional, Callable
from backtest.base import BaseStrategy

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Golden cross / death cross strategy"""
    
    def __init__(fast_period=50, slow_period=200,
                 holding_period=10, price_col='Close')

class RSIStrategy(BaseStrategy):
    """RSI mean reversion strategy"""
    
    def __init__(rsi_period=14, oversold=30,
                 overbought=70, holding_period=10)

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion"""
    
    def __init__(bb_period=20, bb_std=2.0,
                 holding_period=5, price_col='Close')
    
    # Exits early if price crosses middle band

class CustomIndicatorStrategy(BaseStrategy):
    """Flexible strategy using custom functions"""
    
    def __init__(signal_func: Callable,
                 exit_func: Optional[Callable] = None,
                 holding_period: int = 10)
    
    # Example signal_func:
    # lambda df, idx: 'long' if df.iloc[idx]['RSI'] < 30 else None