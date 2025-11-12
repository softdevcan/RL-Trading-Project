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

        # Action space: continuous [-1, +1] for each stock (will be scaled to shares)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32
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
            action: Array of continuous values [-1, 1] for each stock

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # CRITICAL FIX: Flatten action if it comes from VectorizedEnv
        # VectorizedEnv might send shape (1, n_stocks) but we need (n_stocks,)
        action = np.array(action).flatten()

        # Ensure action has correct length
        if len(action) != self.n_stocks:
            logger.error(f"Action length mismatch! Expected {self.n_stocks}, got {len(action)}")
            logger.error(f"Action shape: {np.array(action).shape}, action: {action}")
            logger.error(f"Symbols: {self.symbols}")
            raise ValueError(f"Action must have length {self.n_stocks}, got {len(action)}")

        # Scale continuous actions [-1, 1] to integer shares
        # Use np.round() instead of astype(int) to avoid truncating small actions to zero
        scaled_action = np.round(action * self.max_shares_per_trade).astype(int)
        scaled_action = np.clip(scaled_action, -self.max_shares_per_trade, self.max_shares_per_trade)

        # THRESHOLD REMOVED FOR TESTING!
        # Let ANY non-zero action through to see if model is producing actions
        # Previously: min_threshold = 1 share (1% of 100)
        # Now: NO THRESHOLD - even 1 share will execute
        min_threshold = 0  # DISABLED!
        # for i in range(len(scaled_action)):
        #     if abs(scaled_action[i]) < min_threshold:
        #         scaled_action[i] = 0

        # Önceki portfolio değeri
        prev_portfolio_value = self._get_portfolio_value()

        # İşlemleri gerçekleştir
        trades_executed = 0
        total_commission_cost = 0.0

        for i, shares_to_trade in enumerate(scaled_action):
            if shares_to_trade != 0:
                success, commission = self._execute_trade(i, shares_to_trade)
                if success:
                    trades_executed += 1
                    total_commission_cost += commission

        # Debug logging (reduced frequency - only every 100 steps)
        if self.current_step % 100 == 0:
            logger.info(f"[ENV {id(self)}] Step {self.current_step}: Trades={len(self.trades_history)}, "
                       f"Balance=₺{self.balance:,.0f}, Portfolio=₺{self._get_portfolio_value():,.0f}")

        # Bir adım ilerle
        self.current_step += 1

        # Yeni portfolio değeri
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(current_portfolio_value)

        # Reward hesapla (Faz 1: portfolio value değişimi - komisyon cezası)
        # Portfolio change as percentage
        portfolio_change_pct = ((current_portfolio_value - prev_portfolio_value) / prev_portfolio_value) * 100

        # Commission penalty as percentage of initial balance
        commission_penalty = (total_commission_cost / self.initial_balance) * 100

        # Final reward: portfolio gain minus commission cost
        reward = portfolio_change_pct - commission_penalty

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
            'action': action,
            'trades_executed': trades_executed
        }

        return obs, reward, terminated, truncated, info

    def _execute_trade(self, stock_idx: int, shares: int) -> tuple:
        """
        İşlem gerçekleştir

        Args:
            stock_idx: Hisse index'i
            shares: + (buy) veya - (sell)

        Returns:
            tuple: (success: bool, commission_cost: float)
        """
        symbol = self.symbols[stock_idx]
        current_price = self._get_current_price(symbol)

        if shares > 0:  # BUY
            commission_cost = shares * current_price * self.commission_rate
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
                    'cost': cost,
                    'commission': commission_cost
                })
                return True, commission_cost

        elif shares < 0:  # SELL
            shares_to_sell = min(abs(shares), self.shares_owned[stock_idx])

            if shares_to_sell > 0:
                commission_cost = shares_to_sell * current_price * self.commission_rate
                revenue = shares_to_sell * current_price * (1 - self.commission_rate)
                self.balance += revenue
                self.shares_owned[stock_idx] -= shares_to_sell

                self.trades_history.append({
                    'date': self._get_current_date(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue,
                    'commission': commission_cost
                })
                return True, commission_cost

        return False, 0.0  # İşlem yapılamadı

    def _get_observation(self) -> np.ndarray:
        """
        Current state'i döndür

        State: [balance, shares_owned[N], features[N*10]]
        Features per stock: open, high, low, close, volume, macd, rsi, cci, adx, turbulence
        """
        # Balance (normalize with log scale for better range)
        balance_norm = np.log(self.balance / self.initial_balance + 1)

        # Shares owned (normalize relative to max possible)
        shares_norm = self.shares_owned / self.max_shares_per_trade

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

                # Improved normalization for better neural network learning
                # Use log-scale for volume to handle large variance
                volume_norm = np.log(volume / 1e6 + 1) / 3 if volume > 0 else 0

                features.extend([
                    (open_price - 50) / 50,      # Center around typical BIST price
                    (high_price - 50) / 50,
                    (low_price - 50) / 50,
                    (close_price - 50) / 50,
                    volume_norm,                  # Log-scaled volume
                    np.tanh(macd / 0.1),         # Bounded to [-1, 1]
                    (rsi - 50) / 50,             # Center RSI around 50
                    np.tanh(cci / 100),          # Bounded CCI
                    (adx - 25) / 25,             # Center ADX around 25
                    np.tanh(turbulence / 2)      # Bounded turbulence
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
            print(f"Balance: ₺{self.balance:,.2f}")
            print(f"Portfolio Value: ₺{portfolio_value:,.2f}")
            print(f"Shares Owned: {dict(zip(self.symbols, self.shares_owned.astype(int)))}")

    def get_metrics(self) -> Dict:
        """Trading metriklerini hesapla"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"get_metrics() called: trades_history has {len(self.trades_history)} trades")
        logger.info(f"get_metrics() called: portfolio_values has {len(self.portfolio_values)} values")

        portfolio_values = np.array(self.portfolio_values)

        # Cumulative return
        cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Sharpe ratio (basit hesaplama)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252) if len(returns) > 0 else np.nan

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
    print(f"Initial portfolio value: ₺{info['portfolio_value']:,.2f}")

    # Birkaç random step
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ₺{info['portfolio_value']:,.2f}")

        if terminated:
            break

    # Metrics
    metrics = env.get_metrics()
    print("\n" + "="*60)
    print("EPISODE METRICS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
