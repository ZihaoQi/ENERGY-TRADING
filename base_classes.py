from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import pandas as pd


@dataclass
class Position:
    """Trading position with entry/exit tracking"""
    entry_idx: int
    entry_date: pd.Timestamp
    entry_price: float
    direction: str  # 'long' or 'short'
    metadata: dict[str, Any]

    # Exit data (Optional)
    exit_idx: Optional[int] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    return_pct: Optional[float] = None
    pnl: Optional[float] = None

    def close(self, exit_idx: int, exit_date: pd.Timestamp, exit_price: float) -> None:
        """Close the position and compute return"""
        self.exit_idx = exit_idx
        self.exit_date = exit_date
        self.exit_price = exit_price

        if self.direction == "long":
            self.return_pct = (exit_price - self.entry_price) / self.entry_price * 100
        elif self.direction == "short":
            self.return_pct = (self.entry_price - exit_price) / self.entry_price * 100
        else:
            raise ValueError("Direction must be 'long' or 'short'")

        self.pnl = self.return_pct  # Assuming 100% allocation

    def to_dict(self) -> dict:
        """Convert position to dictionary"""
        return {
            'entry_idx': self.entry_idx,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'exit_idx': self.exit_idx,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'return_pct': self.return_pct,
            'pnl': self.pnl,
            **self.metadata
        }

    @property
    def is_open(self) -> bool:
        """Check if position is still open"""
        return self.exit_idx is None


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, idx: int) -> list[dict[str, Any]]:
        """
        Generate trading signals for the current bar
        
        Args:
            df: DataFrame with market data
            idx: Current index in the dataframe
            
        Returns:
            List of signal dictionaries with keys:
                - direction: 'long' or 'short'
                - metadata: Optional dict with additional signal info
        """
        pass
    
    @abstractmethod
    def should_close_position(self, position: Position, df: pd.DataFrame, 
                             current_idx: int) -> bool:
        """
        Determine if a position should be closed
        
        Args:
            position: The open position
            df: DataFrame with market data
            current_idx: Current index in the dataframe
            
        Returns:
            True if position should be closed, False otherwise
        """
        pass

    # Optional callbacks
    def on_position_opened(self, position: Position, df: pd.DataFrame) -> None:
        pass

    def on_position_closed(self, position: Position, df: pd.DataFrame) -> None:
        pass


class BaseModel(ABC):

    def __init__(self, config: dict = None):
        """
        Args:
            config: config dict containing model parameters
            
        """
        self.config = config or {}

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def get_feature_importance(self):
        return None
