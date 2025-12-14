"""
Benchmark Trading Strategies
Implements Buy-and-Hold and BIST-30 Index strategies for comparison with RL models
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import yfinance as yf
from datetime import datetime, timedelta


class BuyAndHoldStrategy:
    """
    Buy-and-Hold Strategy Implementation

    Buys equal-weighted portfolio at start and holds until end.
    No rebalancing, no trading during the period.
    """

    def __init__(self, initial_balance: float = 1_000_000,
                 commission_rate: float = 0.001):
        """
        Args:
            initial_balance: Starting capital
            commission_rate: Transaction commission rate
        """
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.shares_owned = {}
        self.balance = initial_balance

    def execute(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict]:
        """
        Execute buy-and-hold strategy

        Args:
            df: Multi-index DataFrame with (date, symbol) index

        Returns:
            Tuple of (portfolio_values, returns, trades, metrics)
        """
        # Get unique dates and symbols
        dates = df.index.get_level_values('date').unique().sort_values()
        symbols = df.index.get_level_values('symbol').unique()

        portfolio_values = []
        returns_list = []
        trades = []

        # Initial purchase at first date
        first_date = dates[0]
        first_day_data = df.xs(first_date, level='date')

        # Calculate equal weight allocation
        allocation_per_stock = self.initial_balance / len(symbols)

        # Buy stocks at first day
        for symbol in symbols:
            if symbol not in first_day_data.index:
                continue

            price = first_day_data.loc[symbol, 'close']
            shares_to_buy = int(allocation_per_stock / price)

            if shares_to_buy > 0:
                cost = shares_to_buy * price
                commission = cost * self.commission_rate
                total_cost = cost + commission

                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.shares_owned[symbol] = shares_to_buy

                    trades.append({
                        'date': first_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': price,
                        'total': cost,
                        'commission': commission,
                        'pnl': 0
                    })

        # Track portfolio value over time (but don't trade)
        for i, date in enumerate(dates):
            day_data = df.xs(date, level='date')

            # Calculate portfolio value
            holdings_value = 0.0
            for symbol, shares in self.shares_owned.items():
                if symbol in day_data.index:
                    price = day_data.loc[symbol, 'close']
                    holdings_value += shares * price

            total_value = self.balance + holdings_value
            portfolio_values.append(total_value)

            # Calculate daily return
            if i > 0:
                daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns_list.append(daily_return)

        portfolio_values = np.array(portfolio_values)
        returns = np.array(returns_list)

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_values, returns, trades)

        return portfolio_values, returns, trades, metrics

    def _calculate_metrics(self, portfolio_values: np.ndarray,
                          returns: np.ndarray, trades: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Annualized metrics
        n_days = len(returns)
        annualization_factor = np.sqrt(252)

        mean_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 0
        annualized_return = mean_return * 252

        # Sharpe Ratio
        risk_free_rate = 0.02
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * annualization_factor) if len(excess_returns) > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0] if len(returns) > 0 else np.array([])
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = (np.mean(excess_returns) / (downside_deviation + 1e-10) * annualization_factor) if len(excess_returns) > 0 else 0

        # Maximum Drawdown
        cumulative = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative) / cumulative
        max_drawdown = np.min(drawdown)

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': len(trades),
            'final_portfolio_value': float(portfolio_values[-1]),
            'volatility': float(std_return),
            'n_days': n_days
        }


class BIST30IndexStrategy:
    """
    BIST-30 Index Strategy

    Tracks the BIST-30 index performance using XU030.IS
    If XU030 data is unavailable, creates synthetic index from constituent stocks
    """

    def __init__(self, initial_balance: float = 1_000_000):
        """
        Args:
            initial_balance: Starting capital
        """
        self.initial_balance = initial_balance

    def execute(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict]:
        """
        Execute BIST-30 index tracking strategy

        Args:
            df: Multi-index DataFrame with (date, symbol) index

        Returns:
            Tuple of (portfolio_values, returns, trades, metrics)
        """
        # Get unique dates
        dates = df.index.get_level_values('date').unique().sort_values()

        # Try to fetch XU030 index data
        index_data = self._fetch_index_data(dates[0], dates[-1])

        if index_data is not None:
            # Use actual index data
            portfolio_values, returns = self._track_actual_index(index_data, dates)
        else:
            # Create synthetic index from constituent stocks
            portfolio_values, returns = self._create_synthetic_index(df, dates)

        # BIST-30 index has no trades (passive index)
        trades = []

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_values, returns, trades)

        return portfolio_values, returns, trades, metrics

    def _fetch_index_data(self, start_date, end_date) -> pd.DataFrame:
        """Try to fetch XU030 or XU100 index data from Yahoo Finance"""
        try:
            # Add buffer for data availability
            start = start_date - timedelta(days=7)
            end = end_date + timedelta(days=1)

            # Try multiple index symbols
            index_symbols = ['XU030.IS', 'XU100.IS', '^XU030', '^XU100']

            for symbol in index_symbols:
                try:
                    index = yf.download(symbol, start=start, end=end, progress=False)
                    if not index.empty:
                        return index
                except:
                    continue

            return None
        except:
            return None

    def _track_actual_index(self, index_data: pd.DataFrame,
                           dates: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Track actual BIST-30 index"""
        portfolio_values = []

        # Get first available price
        first_price = index_data['Close'].iloc[0]

        for date in dates:
            # Find closest date in index data
            try:
                if date in index_data.index:
                    current_price = index_data.loc[date, 'Close']
                else:
                    # Use forward fill for missing dates
                    current_price = index_data['Close'].asof(date)
                    if pd.isna(current_price):
                        current_price = first_price

                # Calculate normalized portfolio value
                value = self.initial_balance * (current_price / first_price)
                portfolio_values.append(value)
            except:
                # If date not found, use last known value
                if portfolio_values:
                    portfolio_values.append(portfolio_values[-1])
                else:
                    portfolio_values.append(self.initial_balance)

        portfolio_values = np.array(portfolio_values)

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        return portfolio_values, returns

    def _create_synthetic_index(self, df: pd.DataFrame,
                                dates: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic index from equal-weighted portfolio"""
        symbols = df.index.get_level_values('symbol').unique()
        portfolio_values = []

        # Calculate equal-weighted index
        first_date = dates[0]
        first_day_data = df.xs(first_date, level='date')

        for date in dates:
            try:
                day_data = df.xs(date, level='date')

                # Calculate equal-weighted average return from first day
                total_return = 0
                valid_stocks = 0

                for symbol in symbols:
                    if symbol in day_data.index and symbol in first_day_data.index:
                        first_price = first_day_data.loc[symbol, 'close']
                        current_price = day_data.loc[symbol, 'close']
                        stock_return = (current_price / first_price) - 1
                        total_return += stock_return
                        valid_stocks += 1

                avg_return = total_return / valid_stocks if valid_stocks > 0 else 0
                value = self.initial_balance * (1 + avg_return)
                portfolio_values.append(value)
            except:
                if portfolio_values:
                    portfolio_values.append(portfolio_values[-1])
                else:
                    portfolio_values.append(self.initial_balance)

        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        return portfolio_values, returns

    def _calculate_metrics(self, portfolio_values: np.ndarray,
                          returns: np.ndarray, trades: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Annualized metrics
        n_days = len(returns)
        annualization_factor = np.sqrt(252)

        mean_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 0
        annualized_return = mean_return * 252

        # Sharpe Ratio
        risk_free_rate = 0.02
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * annualization_factor) if len(excess_returns) > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0] if len(returns) > 0 else np.array([])
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = (np.mean(excess_returns) / (downside_deviation + 1e-10) * annualization_factor) if len(excess_returns) > 0 else 0

        # Maximum Drawdown
        cumulative = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative) / cumulative
        max_drawdown = np.min(drawdown)

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': 1,  # Index tracking has minimal trading
            'final_portfolio_value': float(portfolio_values[-1]),
            'volatility': float(std_return),
            'n_days': n_days
        }


def evaluate_benchmarks(test_df: pd.DataFrame,
                       initial_balance: float = 1_000_000,
                       commission_rate: float = 0.001) -> Dict[str, Tuple]:
    """
    Evaluate both benchmark strategies on test data

    Args:
        test_df: Test DataFrame with multi-index (date, symbol)
        initial_balance: Starting capital
        commission_rate: Transaction commission rate

    Returns:
        Dictionary with benchmark results:
        {
            'Buy-and-Hold': (portfolio_values, returns, trades, metrics),
            'BIST-30': (portfolio_values, returns, trades, metrics)
        }
    """
    results = {}

    # 1. Buy-and-Hold Strategy
    print("Evaluating Buy-and-Hold strategy...")
    bah_strategy = BuyAndHoldStrategy(initial_balance, commission_rate)
    bah_portfolio, bah_returns, bah_trades, bah_metrics = bah_strategy.execute(test_df)
    results['Buy-and-Hold'] = (bah_portfolio, bah_returns, bah_trades, bah_metrics)
    print(f"[OK] Buy-and-Hold: Return={bah_metrics['total_return']*100:.2f}%, "
          f"Sharpe={bah_metrics['sharpe_ratio']:.4f}")

    # 2. BIST-30 Index Strategy
    print("Evaluating BIST-30 Index strategy...")
    index_strategy = BIST30IndexStrategy(initial_balance)
    index_portfolio, index_returns, index_trades, index_metrics = index_strategy.execute(test_df)
    results['BIST-30'] = (index_portfolio, index_returns, index_trades, index_metrics)
    print(f"[OK] BIST-30 Index: Return={index_metrics['total_return']*100:.2f}%, "
          f"Sharpe={index_metrics['sharpe_ratio']:.4f}")

    return results


if __name__ == "__main__":
    """Test benchmark strategies"""
    import pickle
    from pathlib import Path

    # Load test data
    data_file = 'data/preprocessed/train_val_test_data.pkl'

    if Path(data_file).exists():
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        test_df = data['test']

        print("=" * 80)
        print("BENCHMARK STRATEGIES EVALUATION")
        print("=" * 80)
        print(f"Test data: {len(test_df)} samples")
        print()

        # Evaluate benchmarks
        benchmark_results = evaluate_benchmarks(test_df)

        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)

        for name, (portfolio, returns, trades, metrics) in benchmark_results.items():
            print(f"\n{name}:")
            print(f"  Total Return: {metrics['total_return']*100:.2f}%")
            print(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Final Value: ${metrics['final_portfolio_value']:,.0f}")
    else:
        print(f"[ERROR] Test data not found: {data_file}")
        print("Please run data preprocessing first!")
