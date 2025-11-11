"""
Gymnasium Trading Environment
Ansari et al. (2024) metodolojisine göre multi-stock trading environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Multi-Stock Trading Environment (Faz 1: Basit versiyon)

    State: [balance, shares_owned[N], OHLCV[N], technicals[N]]
    Action: [-100, ..., 0, ..., +100] per stock (shares to buy/sell)
    Reward: Portfolio value change (Faz 1'de basitleştirilmiş)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        commission_rate: float = 0.001,  # %0.1
        max_shares_per_trade: int = 100,
        frame_bound: Tuple[int, int] = None
    ):
        """
        Args:
            df: Multi-index DataFrame (symbol, date) with OHLCV + indicators
            initial_balance: Starting cash
            commission_rate: Transaction cost (default 0.1%)
            max_shares_per_trade: Maximum shares to trade per action
            frame_bound: (start_idx, end_idx) for episode
        """
        super().__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_shares_per_trade = max_shares_per_trade

        # Semboller
        self.symbols = df.index.get_level_values('symbol').unique().tolist()
        self.n_stocks = len(self.symbols)

        # Tarihleri al
        self.dates = df.index.get_level_values('date').unique().sort_values()

        # Frame bound ayarla
        if frame_bound is None:
            self.frame_bound = (0, len(self.dates) - 1)
        else:
            self.frame_bound = frame_bound

        # Feature sayısı (Faz 1: sadece OHLCV + technical indicators)
        # Features per stock: open, high, low, close, volume + 5 technicals = 10
        self.features_per_stock = 10

        # State space: balance (1) + shares_owned (N) + features (N * features_per_stock)
        state_dim = 1 + self.n_stocks + (self.n_stocks * self.features_per_stock)

        # Observation space: continuous
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Action space: [-max_shares, +max_shares] for each stock
        self.action_space = spaces.Box(
            low=-max_shares_per_trade,
            high=max_shares_per_trade,
            shape=(self.n_stocks,),
            dtype=np.int32
        )

        # Trading variables
        self.current_step = None
        self.balance = None
        self.shares_owned = None
        self.portfolio_values = []
        self.trades_history = []

        logger.info(f"TradingEnv initialized:")
        logger.info(f"  Stocks: {self.n_stocks} ({', '.join(self.symbols)})")
        logger.info(f"  Trading days: {self.frame_bound[1] - self.frame_bound[0] + 1}")
        logger.info(f"  State dimension: {state_dim}")
        logger.info(f"  Action dimension: {self.n_stocks}")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Environment'ı başlangıç durumuna döndür"""
        super().reset(seed=seed)

        self.current_step = self.frame_bound[0]
        self.balance = self.initial_balance
        self.shares_owned = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_balance]
        self.trades_history = []

        state = self._get_observation()

        info = {
            'date': self._get_current_date(),
            'balance': self.balance,
            'portfolio_value': self._get_portfolio_value()
        }

        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Bir adım ilerle

        Args:
            action: Array of shares to buy (+) or sell (-) for each stock

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Action'ı integer'a çevir ve clip et
        action = np.clip(action.astype(int), -self.max_shares_per_trade, self.max_shares_per_trade)

        # Önceki portfolio değeri
        prev_portfolio_value = self._get_portfolio_value()

        # İşlemleri gerçekleştir
        for i, shares_to_trade in enumerate(action):
            if shares_to_trade != 0:
                self._execute_trade(i, shares_to_trade)

        # Bir adım ilerle
        self.current_step += 1

        # Yeni portfolio değeri
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(current_portfolio_value)

        # Reward hesapla (Faz 1: basit - portfolio value değişimi)
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # Episode sonu kontrolü
        terminated = self.current_step >= self.frame_bound[1]
        truncated = False

        # Observation
        obs = self._get_observation()

        # Info
        info = {
            'date': self._get_current_date(),
            'balance': self.balance,
            'portfolio_value': current_portfolio_value,
            'shares_owned': self.shares_owned.copy(),
            'action': action
        }

        return obs, reward, terminated, truncated, info

    def _execute_trade(self, stock_idx: int, shares: int):
        """
        İşlem gerçekleştir

        Args:
            stock_idx: Hisse index'i
            shares: + (buy) veya - (sell)
        """
        symbol = self.symbols[stock_idx]
        current_price = self._get_current_price(symbol)

        if shares > 0:  # BUY
            cost = shares * current_price * (1 + self.commission_rate)

            if cost <= self.balance:
                self.balance -= cost
                self.shares_owned[stock_idx] += shares

                self.trades_history.append({
                    'date': self._get_current_date(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'cost': cost
                })

        elif shares < 0:  # SELL
            shares_to_sell = min(abs(shares), self.shares_owned[stock_idx])

            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.commission_rate)
                self.balance += revenue
                self.shares_owned[stock_idx] -= shares_to_sell

                self.trades_history.append({
                    'date': self._get_current_date(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue
                })

    def _get_observation(self) -> np.ndarray:
        """
        Current state'i döndür

        State: [balance, shares_owned[N], features[N*10]]
        Features per stock: open, high, low, close, volume, macd, rsi, cci, adx, turbulence
        """
        # Balance (normalize)
        balance_norm = self.balance / self.initial_balance

        # Shares owned (normalize)
        shares_norm = self.shares_owned / 100  # Rough normalization

        # Market features
        features = []
        current_date = self._get_current_date()

        for symbol in self.symbols:
            # Hissenin o günkü verisini al
            try:
                row = self.df.loc[(symbol, current_date)]

                # OHLCV
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                volume = row['volume']

                # Technical indicators
                macd = row.get('macd', 0)
                rsi = row.get('rsi', 50)
                cci = row.get('cci', 0)
                adx = row.get('adx', 0)
                turbulence = row.get('turbulence', 0)

                # Normalize (basit normalizasyon)
                features.extend([
                    open_price / 100,  # Rough normalization
                    high_price / 100,
                    low_price / 100,
                    close_price / 100,
                    volume / 1e6,
                    macd / 10,
                    rsi / 100,
                    cci / 100,
                    adx / 100,
                    turbulence / 10
                ])

            except KeyError:
                # Veri yoksa 0 ile doldur
                features.extend([0] * self.features_per_stock)

        # State vector oluştur
        state = np.concatenate([
            [balance_norm],
            shares_norm,
            features
        ]).astype(np.float32)

        return state

    def _get_current_price(self, symbol: str) -> float:
        """Hissenin güncel fiyatını döndür (close price)"""
        current_date = self._get_current_date()
        try:
            return self.df.loc[(symbol, current_date), 'close']
        except KeyError:
            logger.warning(f"Price not found for {symbol} on {current_date}")
            return 0.0

    def _get_current_date(self):
        """Güncel tarihi döndür"""
        return self.dates[self.current_step]

    def _get_portfolio_value(self) -> float:
        """Toplam portfolio değeri (balance + stock values)"""
        stock_value = 0.0

        for i, symbol in enumerate(self.symbols):
            if self.shares_owned[i] > 0:
                current_price = self._get_current_price(symbol)
                stock_value += self.shares_owned[i] * current_price

        return self.balance + stock_value

    def render(self, mode='human'):
        """Environment'ı görselleştir"""
        if mode == 'human':
            portfolio_value = self._get_portfolio_value()
            print(f"\nStep: {self.current_step}")
            print(f"Date: {self._get_current_date()}")
            print(f"Balance: ${self.balance:,.2f}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Shares Owned: {dict(zip(self.symbols, self.shares_owned.astype(int)))}")

    def get_metrics(self) -> Dict:
        """Trading metriklerini hesapla"""
        portfolio_values = np.array(self.portfolio_values)

        # Cumulative return
        cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Sharpe ratio (basit hesaplama)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)

        # Max drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown)

        return {
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': len(self.trades_history)
        }


def make_env(df: pd.DataFrame, **kwargs) -> TradingEnv:
    """Environment factory function"""
    return TradingEnv(df, **kwargs)


if __name__ == '__main__':
    # Test
    from data.data_fetcher import DataFetcher
    from data.bist30_symbols import get_symbols
    from data.technical_indicators import add_indicators_to_multi_symbol_df

    logger.setLevel(logging.INFO)

    # Veri yükle veya çek
    symbols = get_symbols(phase=1)
    fetcher = DataFetcher()

    try:
        df = fetcher.load_data('stock_data_with_indicators.csv')
    except:
        print("Data not found, fetching...")
        df = fetcher.fetch_stock_data(symbols)
        df = fetcher.clean_data(df)
        df = add_indicators_to_multi_symbol_df(df)
        fetcher.save_data(df, 'stock_data_with_indicators.csv')

    # Train split
    train_df, val_df, test_df = fetcher.split_data(df)

    # Environment oluştur
    env = make_env(train_df, initial_balance=1_000_000)

    print("\n" + "="*60)
    print("TRADING ENVIRONMENT TEST")
    print("="*60)

    # Reset
    obs, info = env.reset()
    print(f"\nInitial state shape: {obs.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:,.2f}")

    # Birkaç random step
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")

        if terminated:
            break

    # Metrics
    metrics = env.get_metrics()
    print("\n" + "="*60)
    print("EPISODE METRICS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
