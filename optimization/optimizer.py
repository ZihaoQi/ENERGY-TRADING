import itertools
import pandas as pd
from backtest.backtester import Backtester, PerformanceAnalyzer


class StrategyOptimizer:
    """
    Generic grid-search optimizer for trading strategies.
    """

    def __init__(
        self,
        strategy_class,
        df: pd.DataFrame,
        fixed_params: dict,
        param_grid: dict[str, list],
        price_col: str = "Close",
        commission: float = 0.0,
        slippage: float = 0.0,
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            strategy_class: Strategy class (not instance!)
            df: Market dataframe
            fixed_params: Parameters that stay constant
            param_grid: Dict of parameters to optimize
            metric: Metric name from PerformanceAnalyzer
            maximize: True if higher metric is better
        """

        self.strategy_class = strategy_class
        self.df = df
        self.fixed_params = fixed_params
        self.param_grid = param_grid
        self.price_col = price_col
        self.commission = commission
        self.slippage = slippage
        self.metric = metric
        self.maximize = maximize
        self.verbose = verbose

        self.results = []

    # ======================================================
    # Public API
    # ======================================================

    def run(self) -> pd.DataFrame:
        """
        Run grid search optimization.
        Returns dataframe of all parameter results.
        """

        keys = list(self.param_grid.keys())
        combinations = list(itertools.product(*self.param_grid.values()))

        best_score = float("-inf") if self.maximize else float("inf")
        best_params = None

        if self.verbose:
            print(f"\nRunning optimization over {len(combinations)} combinations...")

        for values in combinations:
            params = dict(zip(keys, values))
            full_params = {**self.fixed_params, **params}

            score = self._evaluate(full_params)

            row = {**params, self.metric: score}
            self.results.append(row)

            if self.verbose:
                print(f"Params: {params} → {self.metric}: {score:.4f}")

            if score is None:
                continue

            if self.maximize and score > best_score:
                best_score = score
                best_params = params

            elif not self.maximize and score < best_score:
                best_score = score
                best_params = params

        if self.verbose:
            print("\n✓ Optimization complete")
            print(f"Best params: {best_params}")
            print(f"Best {self.metric}: {best_score:.4f}")

        return pd.DataFrame(self.results)

    # ======================================================
    # Internal
    # ======================================================

    def _evaluate(self, params: dict) -> float:
        """
        Instantiate strategy, run backtest, return metric score.
        """

        strategy = self.strategy_class(**params)

        backtester = Backtester(
            df=self.df,
            strategy=strategy,
            price_col=self.price_col,
            commission=self.commission,
            slippage=self.slippage,
            verbose=False,
        )

        trades = backtester.run_backtest()

        analyzer = PerformanceAnalyzer(
            pd.DataFrame(trades),
            verbose=False
        )

        results = analyzer.calculate_metrics()

        if results is None:
            return None

        return results["metrics"].get(self.metric, None)
