from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from base_classes import Position, BaseStrategy


class Backtester:
    """Core backtesting engine"""

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        price_col: str = "Close",
        max_positions: int = 1,
        commission: float = 0.0,
        slippage: float = 0.0,
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.strategy = strategy
        self.price_col = price_col
        self.max_positions = max_positions
        self.commission = commission
        self.slippage = slippage
        self.verbose = verbose

        self.trades: list[dict] = []
        self.positions: list[Position] = []

        self._validate_data()

    # ==========================================================
    # Public API
    # ==========================================================

    def run_backtest(self) -> list[dict]:
        """Execute backtest and return completed trades"""

        if self.verbose:
            print("\n" + "=" * 60)
            print("RUNNING BACKTEST")
            print("=" * 60)

        for i in range(len(self.df)):
            current_date = self.df.iloc[i]["Date"]

            # 1️⃣ Exit first
            self._check_position_exits(i, current_date)

            # 2️⃣ Then check entries
            if len(self.positions) < self.max_positions:
                self._check_new_signals(i, current_date)

        # Force close remaining positions
        for pos in self.positions[:]:
            self._close_position(
                position=pos,
                idx=len(self.df) - 1,
                date=self.df.iloc[-1]["Date"],
                forced=True,
            )

        if self.verbose:
            print(f"\n✓ Backtest complete. Total trades: {len(self.trades)}")

        return self.trades

    def get_open_positions(self) -> list[Position]:
        """Return current open positions"""
        return self.positions.copy()

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame"""
        return pd.DataFrame(self.trades)

    # ==========================================================
    # Internal Methods
    # ==========================================================

    def _validate_data(self):
        """Validate input dataframe"""
        if "Date" not in self.df.columns:
            raise ValueError("DataFrame must contain 'Date' column")

        if self.price_col not in self.df.columns:
            raise ValueError(f"Price column '{self.price_col}' not found")

        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date").reset_index(drop=True)

    def _check_new_signals(self, i: int, date: pd.Timestamp):
        """Check strategy for new signals"""
        signals = self.strategy.generate_signals(self.df, i)

        for signal in signals:
            if len(self.positions) >= self.max_positions:
                break

            self._open_position(
                i=i,
                date=date,
                direction=signal["direction"],
                metadata=signal.get("metadata", {}),
            )

    def _open_position(
        self,
        i: int,
        date: pd.Timestamp,
        direction: str,
        metadata: Optional[dict] = None,
    ):
        """Open new position with slippage-adjusted fill"""

        raw_price = self.df.iloc[i][self.price_col]
        fill_price = self._calculate_fill_price(
            price=raw_price,
            direction=direction,
            is_entry=True,
        )

        position = Position(
            entry_idx=i,
            entry_date=date,
            entry_price=fill_price,
            direction=direction,
            metadata=metadata,
        )

        self.positions.append(position)

        # if self.verbose:
        #     print(f"[OPEN] {direction.upper()} @ {fill_price:.4f} ({date})")

    def _check_position_exits(self, idx: int, date: pd.Timestamp):
        """Check open positions for exit signals"""

        to_close = []

        for pos in self.positions:
            if self.strategy.should_close_position(pos, self.df, idx):
                to_close.append(pos)

        for pos in to_close:
            self._close_position(pos, idx, date)

    def _close_position(
        self,
        position: Position,
        idx: int,
        date: pd.Timestamp,
        forced: bool = False,
    ):
        """Close position and record trade"""

        raw_price = self.df.iloc[idx][self.price_col]
        fill_price = self._calculate_fill_price(
            price=raw_price,
            direction=position.direction,
            is_entry=False,
        )

        position.close(
            exit_idx=idx,
            exit_date=date,
            exit_price=fill_price,
        )

        trade = position.to_dict()

        # Apply commission (round trip)
        total_commission = self.commission * 2
        trade["commission"] = total_commission

        if "pnl" in trade:
            trade["pnl"] -= total_commission

        if forced:
            trade["forced_exit"] = True

        self.trades.append(trade)
        self.positions.remove(position)

        # if self.verbose:
        #     print(f"[CLOSE] {position.direction.upper()} @ {fill_price:.4f} ({date})")

    def _calculate_fill_price(
        self,
        price: float,
        direction: str,
        is_entry: bool,
    ) -> float:
        """
        Adjust price for slippage.
        """

        if self.slippage == 0:
            return price
        
        direction = direction.lower()

        if direction == "long":
            return price + self.slippage if is_entry else price - self.slippage

        elif direction == "short":
            return price - self.slippage if is_entry else price + self.slippage

        else:
            raise ValueError("Direction must be 'long' or 'short'")


class PerformanceAnalyzer:
    """Calculate performance metrics"""

    def __init__(self, trades_df: pd.DataFrame, verbose: bool = True):
        self.verbose = verbose

        if trades_df is None or trades_df.empty:
            self.trades_df = None
        else:
            self.trades_df = trades_df.copy()
            self.trades_df["entry_date"] = pd.to_datetime(self.trades_df["entry_date"])
            self.trades_df["exit_date"] = pd.to_datetime(self.trades_df["exit_date"])
            self.trades_df.sort_values("exit_date", inplace=True)

    # ==========================================================
    # Public API
    # ==========================================================
    def calculate_metrics(self) -> Optional[dict]:

        if self.trades_df is None:
            if self.verbose:
                print("No trades executed!")
            return None

        basic_metrics = self._calculate_basic_metrics()
        risk_metrics = self._calculate_risk_metrics()

        monthly_returns = self._calculate_monthly_returns()
        annual_returns = self._calculate_annual_returns()

        all_metrics = {**basic_metrics, **risk_metrics}

        if self.verbose:
            self._print_results(all_metrics, annual_returns, monthly_returns)

        return {
            "trades_df": self.trades_df,
            "monthly_returns": monthly_returns,
            "annual_returns": annual_returns,
            "metrics": all_metrics
        }

    # ==========================================================
    # Private Methods
    # ==========================================================
    def _calculate_basic_metrics(self) -> dict:

        returns = self.trades_df["return_pct"]

        total_trades = len(returns)
        winning_trades = (returns > 0).sum()
        losing_trades = (returns < 0).sum()

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        avg_return = returns.mean()
        winning_trades_avg_return = returns[returns > 0].mean() if winning_trades > 0 else 0
        losing_trades_avg_return = returns[returns < 0].mean() if losing_trades > 0 else 0
        max_return = returns.max()
        max_loss = returns.min()

        profit_factor = (
            returns[returns > 0].sum() /
            abs(returns[returns < 0].sum())
            if returns[returns < 0].sum() != 0 else np.nan
        )

        expectancy = returns.mean()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "winning_trades_avg_return": winning_trades_avg_return,
            "losing_trades_avg_return": losing_trades_avg_return,
            "max_return": max_return,
            "max_loss": max_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy
        }

    def _calculate_risk_metrics(self) -> dict:

        returns = self.trades_df["return_pct"] / 100

        if len(returns) < 2:
            return {
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "max_drawdown": np.nan
            }

        # Sharpe Ratio (trade-based)
        sharpe_ratio = (
            returns.mean() / returns.std()
            if returns.std() != 0 else np.nan
        )

        # Sortino Ratio
        downside = returns[returns < 0]
        downside_std = downside.std()

        sortino_ratio = (
            returns.mean() / downside_std
            if downside_std != 0 else np.nan
        )

        # Max Drawdown (on cumulative returns)
        cumulative = returns.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown * 100  # back to %
        }

    def _calculate_monthly_returns(self) -> pd.Series:

        df = self.trades_df.copy()
        df["year_month"] = df["exit_date"].dt.to_period("M")

        return df.groupby("year_month")["return_pct"].sum()

    def _calculate_annual_returns(self) -> pd.Series:

        df = self.trades_df.copy()
        df["year"] = df["exit_date"].dt.year

        return df.groupby("year")["return_pct"].sum()

    def _print_results(self,
                       metrics: dict,
                       annual_returns: pd.Series,
                       monthly_returns: pd.Series):

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nTotal Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Average Return per Trade: {metrics['avg_return']:.2f}%")
        print(f"Average Return per Winning Trade: {metrics['winning_trades_avg_return']:.2f}%")
        print(f"Average Return per Losing Trade: {metrics['losing_trades_avg_return']:.2f}%")
        print(f"Max Return: {metrics['max_return']:.2f}%")
        print(f"Max Loss: {metrics['max_loss']:.2f}%")

        print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"Expectancy: {metrics['expectancy']:.2f}%")

        print("\n" + "-" * 60)
        print("ANNUAL RETURNS")
        print("-" * 60)
        for year, ret in annual_returns.items():
            print(f"{year}: {ret:.2f}%")

        print("\n" + "-" * 60)
        print("MONTHLY RETURNS (Last 12 months)")
        print("-" * 60)
        for month, ret in monthly_returns.tail(12).items():
            print(f"{month}: {ret:.2f}%")



# ==========================================================
# Main 5-Panel Visualization
# ==========================================================
def visualize_backtest_results(results: dict,
                               save_path: str = None,
                               show_plot: bool = False) -> Figure:
    """
    Creates 5-panel backtest visualization:

    1. Cumulative returns with drawdown
    2. Annual returns bar chart
    3. Monthly returns heatmap
    4. Returns distribution histogram
    5. Win/loss pie chart
    """

    trades_df = results["trades_df"]
    monthly_returns = results["monthly_returns"]
    annual_returns = results["annual_returns"]
    metrics = results["metrics"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    _plot_cumulative_returns(fig, gs, trades_df)
    _plot_annual_returns(fig, gs, annual_returns)
    _plot_monthly_heatmap(fig, gs, monthly_returns)
    _plot_returns_distribution(fig, gs, trades_df)
    _plot_win_loss_pie(fig, gs, metrics)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Backtest visualization saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


# ==========================================================
# Equity Curve (Dollar Based)
# ==========================================================
def plot_equity_curve(trades_df: pd.DataFrame,
                      initial_capital: float = 10000,
                      save_path: str = None) -> Figure:
    """
    Plots equity curve in dollar terms.
    """

    trades_df = trades_df.sort_values("exit_date").copy()

    equity = [initial_capital]
    for r in trades_df["return_pct"]:
        equity.append(equity[-1] * (1 + r / 100))

    equity = equity[1:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(trades_df["exit_date"], equity, linewidth=2)
    ax.set_title("Equity Curve ($)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ==========================================================
# Private Helper Functions
# ==========================================================
def _plot_cumulative_returns(fig: Figure, gs, trades_df: pd.DataFrame):
    ax = fig.add_subplot(gs[0, :])

    trades_df = trades_df.sort_values("exit_date")
    cumulative = trades_df["return_pct"].cumsum()

    # Drawdown calculation
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    ax.plot(trades_df["exit_date"], cumulative, linewidth=2)
    ax.fill_between(trades_df["exit_date"], cumulative, alpha=0.3)

    ax.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, alpha=0.3)

    # Optional secondary axis for drawdown
    ax2 = ax.twinx()
    ax2.fill_between(trades_df["exit_date"], drawdown, alpha=0.15)
    ax2.set_ylabel("Drawdown (%)")


def _plot_annual_returns(fig: Figure, gs, annual_returns: pd.Series):
    ax = fig.add_subplot(gs[1, 0])

    colors = ["green" if x > 0 else "red" for x in annual_returns.values]

    ax.bar(annual_returns.index.astype(str),
           annual_returns.values,
           color=colors,
           alpha=0.7)

    ax.axhline(0, linewidth=0.8)
    ax.set_title("Annual Returns", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_monthly_heatmap(fig: Figure, gs, monthly_returns: pd.Series):
    ax = fig.add_subplot(gs[1, 1])

    monthly_df = monthly_returns.reset_index()
    monthly_df["year"] = monthly_df["year_month"].dt.year
    monthly_df["month"] = monthly_df["year_month"].dt.month

    pivot = monthly_df.pivot(index="year",
                             columns="month",
                             values="return_pct")

    sns.heatmap(pivot,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                center=0,
                ax=ax,
                cbar_kws={"label": "Return (%)"})

    ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")


def _plot_returns_distribution(fig: Figure, gs, trades_df: pd.DataFrame):
    ax = fig.add_subplot(gs[2, 0])

    returns = trades_df["return_pct"]

    ax.hist(returns, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, linestyle="--", linewidth=2, label="Break-even")
    ax.axvline(returns.mean(),
               linestyle="--",
               linewidth=2,
               label=f"Mean ({returns.mean():.2f}%)")

    ax.set_title("Trade Returns Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Return per Trade (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


def _plot_win_loss_pie(fig: Figure, gs, metrics: dict):
    ax = fig.add_subplot(gs[2, 1])

    win_loss = [
        metrics["winning_trades"],
        metrics["losing_trades"]
    ]

    ax.pie(win_loss,
           labels=["Wins", "Losses"],
           autopct="%1.1f%%",
           startangle=90)

    ax.set_title(
        f'Win/Loss Ratio (Win Rate: {metrics["win_rate"]:.1f}%)',
        fontsize=14,
        fontweight="bold"
    )
