"""
Gymnasium Trading Environment
Ansari et al. (2024) metodolojisine göre multi-stock trading environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Multi-Stock Trading Environment (Faz 2: Fundamental + Macro data)

    State: [balance, shares_owned[N], OHLCV[N], technicals[N], fundamental[N], macro[6]]
    Action: [-100, ..., 0, ..., +100] per stock (shares to buy/sell)
    Reward: Portfolio value change (Faz 1) or PSR reward (Faz 2)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        commission_rate: float = 0.001,  # %0.1
        max_shares_per_trade: int = 100,
        frame_bound: Optional[Tuple[int, int]] = None,
        fundamental_df: Optional[pd.DataFrame] = None,  # Faz 2: Fundamental data
        macro_df: Optional[pd.DataFrame] = None,         # Faz 2: Macro data
        phase: int = 1,                        # 1=Basit, 2=Fundamental+Macro
        reward_type: str = 'simple',           # 'simple' or 'psr'
        reward_weights: Optional[Dict[str, float]] = None,  # PSR weights (optional)
        prediction_features: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        # Faz 3.3: ATR-tabanli pozisyon boyutlandirma
        use_atr_sizing: bool = False,
        risk_per_trade: float = 0.02,       # Portfolio degerinin %2'si
        atr_multiplier: float = 2.0,        # ATR stop mesafesi katsayisi
        max_position_pct: float = 0.20,     # Tek pozisyonun maks portfolio payi
        atr_period: int = 14,               # ATR hesaplama periyodu
        # Faz 3.3.2: Kelly fraksiyonel boyutlandirma
        use_kelly: bool = False,
        kelly_fraction: float = 0.25,       # Quarter-Kelly (muhafazakar)
    ):
        """
        Args:
            df: Multi-index DataFrame (symbol, date) with OHLCV + indicators
            initial_balance: Starting cash
            commission_rate: Transaction cost (default 0.1%)
            max_shares_per_trade: Maximum shares to trade per action
            frame_bound: (start_idx, end_idx) for episode
            fundamental_df: Fundamental ratios per symbol (Faz 2)
            macro_df: Macro indicators per date (Faz 2)
            phase: 1=Faz1 (56 features), 2=Faz2 (97 features), 3=Faz3 (67 features, +gold)
            reward_type: 'simple' (Faz 1/3) or 'psr' (Faz 2)
            reward_weights: PSR weights dict (w1-w5), None uses defaults
            prediction_features: {date_str: {symbol: {predicted_return, predicted_direction,
                confidence, ensemble_agreement}}} — ensemble tahmin verileri
        """
        super().__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_shares_per_trade = max_shares_per_trade
        self.phase = phase
        self.reward_type = reward_type

        # ATR-tabanlı pozisyon boyutlandirma (Faz 3.3)
        self.use_atr_sizing = use_atr_sizing
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.max_position_pct = max_position_pct
        self.atr_period = atr_period
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction

        # Faz 2 data
        self.fundamental_df = fundamental_df
        self.macro_df = macro_df

        # Prediction features (4 per stock: predicted_return, direction, confidence, agreement)
        self.prediction_features = prediction_features
        self.use_predictions = prediction_features is not None
        self.prediction_features_per_stock = 4

        # Reward calculator instance
        from env.reward_functions import RewardCalculator, SimpleRewardCalculator

        if reward_type == 'psr':
            if reward_weights:
                # Convert float values to int where needed for RewardCalculator
                int_weights = {}
                for k, v in reward_weights.items():
                    if k in ['rolling_window', 'target_trades_per_100']:
                        int_weights[k] = int(v)
                    else:
                        int_weights[k] = v
                self.reward_calculator = RewardCalculator(**int_weights)
            else:
                self.reward_calculator = RewardCalculator()  # Use defaults
            logger.info(f"Using PSR reward with weights: {self.reward_calculator.get_weights()}")
        else:
            self.reward_calculator = SimpleRewardCalculator()
            logger.info("Using Simple reward (baseline)")

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

        # Feature sayısı hesapla (phase'e göre)
        if phase in (1, 3):
            self.features_per_stock = 10
            state_dim = 1 + self.n_stocks + (self.n_stocks * self.features_per_stock)
        else:  # phase == 2
            self.features_per_stock = 17
            self.macro_features = 6
            state_dim = 1 + self.n_stocks + (self.n_stocks * self.features_per_stock) + self.macro_features

        # Prediction features: 4 per stock (predicted_return, direction, confidence, agreement)
        if self.use_predictions:
            state_dim += self.n_stocks * self.prediction_features_per_stock

        # Observation space: bounded for gradient stability (#1)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
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

        # Price normalization stats — per symbol, computed from training data (#2)
        # price_stats[symbol] = {'mean': float, 'std': float}
        self.price_stats: Dict[str, Dict[str, float]] = {}
        self._compute_price_stats()

        # Trading variables
        self.current_step: int = 0
        self.balance: float = initial_balance
        self.shares_owned: np.ndarray = np.zeros(self.n_stocks)
        self.portfolio_values: List[float] = []
        self.trades_history: List[Dict] = []
        self.returns_history: List[float] = []  # For PSR reward

        logger.info(f"TradingEnv initialized (Phase {phase}, Reward: {reward_type.upper()}):")
        logger.info(f"  Stocks: {self.n_stocks} ({', '.join(self.symbols)})")
        logger.info(f"  Trading days: {self.frame_bound[1] - self.frame_bound[0] + 1}")
        logger.info(f"  State dimension: {state_dim}")
        logger.info(f"  Action dimension: {self.n_stocks}")
        if self.use_predictions:
            logger.info(f"  Prediction features: ✓ (+{self.n_stocks * self.prediction_features_per_stock} features)")
        if phase == 2:
            logger.info(f"  Fundamental data: {'✓' if fundamental_df is not None else '✗'}")
            logger.info(f"  Macro data: {'✓' if macro_df is not None else '✗'}")
        elif phase == 3:
            gold_assets = [s for s in self.symbols if s in ('GOLD_GRAM_TRY', 'GC=F')]
            logger.info(f"  Gold assets: {gold_assets if gold_assets else 'none in df'}")

    def _compute_price_stats(self):
        """Compute per-symbol price mean/std from the full df for dynamic normalization (#2)."""
        for symbol in self.symbols:
            try:
                prices = self.df.xs(symbol, level='symbol')['close'].astype(float)
                mean = float(prices.mean())
                std = float(prices.std())
                self.price_stats[symbol] = {
                    'mean': mean,
                    'std': std if std > 1e-8 else 1.0
                }
            except Exception:
                self.price_stats[symbol] = {'mean': 50.0, 'std': 50.0}

    def get_price_stats(self) -> Dict[str, Dict[str, float]]:
        """Return price normalization stats (save these for inference)."""
        return self.price_stats

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Environment'ı başlangıç durumuna döndür"""
        super().reset(seed=seed)

        self.current_step = self.frame_bound[0]
        self.balance = self.initial_balance
        self.shares_owned = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_balance]
        self.trades_history = []
        self.returns_history = []

        # Reset reward calculator state (DSR için gerekli)
        self.reward_calculator.reset()

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
        if self.use_kelly:
            # Kelly fraksiyonel boyutlandirma (Faz 3.3.2)
            current_date_str = str(self._get_current_date())[:10]
            date_preds = (self.prediction_features or {}).get(current_date_str, {})
            scaled_action = np.array([
                self._kelly_position_size(
                    self.symbols[i],
                    float(action[i]),
                    win_prob=float(date_preds.get(self.symbols[i], {}).get('confidence', 0.5 + abs(float(action[i])) * 0.3)),
                )
                for i in range(self.n_stocks)
            ], dtype=int)
        elif self.use_atr_sizing:
            # ATR-tabanli dinamik pozisyon boyutlandirma (Faz 3.3.1)
            scaled_action = np.array([
                self._atr_position_size(self.symbols[i], float(action[i]))
                for i in range(self.n_stocks)
            ], dtype=int)
        else:
            # Sabit lot: orijinal davranis (geriye uyumlu)
            scaled_action = np.round(action * self.max_shares_per_trade).astype(int)
            scaled_action = np.clip(scaled_action, -self.max_shares_per_trade, self.max_shares_per_trade)

        # Minimum trade threshold: filter out near-zero actions (#5)
        min_threshold = 1
        for i in range(len(scaled_action)):
            if abs(scaled_action[i]) < min_threshold:
                scaled_action[i] = 0

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

        # Günlük getiriyi hesapla ve kaydet
        portfolio_return = ((current_portfolio_value - prev_portfolio_value) / prev_portfolio_value) * 100
        self.returns_history.append(portfolio_return)

        # Reward hesapla (reward type'a göre)
        if self.reward_type == 'psr':
            # PSR Reward
            from env.reward_functions import RewardCalculator
            if isinstance(self.reward_calculator, RewardCalculator):
                reward, reward_components = self.reward_calculator.calculate_psr_reward(
                    portfolio_values=self.portfolio_values,
                    returns=self.returns_history,
                    current_step=self.current_step - self.frame_bound[0],  # Relative step
                    trades_executed=len(self.trades_history),
                    commission_cost=total_commission_cost,
                    initial_balance=self.initial_balance
                )
            else:
                reward, reward_components = 0.0, {}
        else:
            # Simple Reward (Faz 1 baseline)
            from env.reward_functions import SimpleRewardCalculator
            if isinstance(self.reward_calculator, SimpleRewardCalculator):
                reward, reward_components = self.reward_calculator.calculate_reward(
                    prev_portfolio_value=prev_portfolio_value,
                    current_portfolio_value=current_portfolio_value,
                    commission_cost=total_commission_cost,
                    initial_balance=self.initial_balance
                )
            else:
                reward, reward_components = 0.0, {}

        # Reward sanity check — the reward calculators already clip and guard,
        # but assert here so any future regression that emits NaN/Inf surfaces
        # immediately rather than silently diverging SB3 training.
        if not np.isfinite(reward):
            logger.error(
                f"Non-finite reward at step {self.current_step}: {reward} "
                f"(components={reward_components})"
            )
            reward = 0.0

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
            'trades_executed': trades_executed,
            'reward_components': reward_components  # PSR breakdown için
        }

        # Episode bittiğinde metrics ekle
        if terminated:
            metrics = self.get_metrics()
            info.update(metrics)

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

        Faz 1: [balance, shares_owned[N], OHLCV+Technical[N*10]]
        Faz 2: [balance, shares_owned[N], OHLCV+Technical+Fundamental[N*17], Macro[6]]
        """
        # Balance (normalize with log scale for better range)
        balance_norm = np.log(float(self.balance) / self.initial_balance + 1)

        # Shares owned (normalize relative to max possible)
        shares_norm = self.shares_owned.astype(np.float32) / self.max_shares_per_trade

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
                # Convert pandas scalar to Python float
                if isinstance(volume, (int, float, np.integer, np.floating)):
                    volume_val = float(volume)
                else:
                    try:
                        volume_val = float(volume)  # type: ignore
                    except:
                        volume_val = 0.0
                volume_norm = np.log(volume_val / 1e6 + 1) / 3 if volume_val > 0 else 0.0

                # Dynamic z-score normalization using per-symbol price stats (#2)
                stats = self.price_stats.get(symbol, {'mean': 50.0, 'std': 50.0})
                p_mean, p_std = stats['mean'], stats['std']

                stock_features = [
                    (float(open_price) - p_mean) / p_std,
                    (float(high_price) - p_mean) / p_std,
                    (float(low_price) - p_mean) / p_std,
                    (float(close_price) - p_mean) / p_std,
                    volume_norm,
                    np.tanh(macd / 0.1),
                    (rsi - 50) / 50,
                    np.tanh(cci / 100),
                    (adx - 25) / 25,
                    np.tanh(turbulence / 2)
                ]

                # Faz 2: Fundamental ratios ekle
                if self.phase == 2 and self.fundamental_df is not None:
                    try:
                        fund_row = self.fundamental_df.loc[symbol]

                        # 7 fundamental ratios (normalized)
                        roe = np.tanh(fund_row.get('roe', 0) / 50)           # ROE normalize
                        roa = np.tanh(fund_row.get('roa', 0) / 20)           # ROA normalize
                        debt_eq = np.tanh(fund_row.get('debt_to_equity', 0) / 100)
                        curr_ratio = (fund_row.get('current_ratio', 1.5) - 1.5) / 2
                        pe_ratio = np.tanh(fund_row.get('pe_ratio', 10) / 30)
                        pb_ratio = np.tanh(fund_row.get('pb_ratio', 2) / 10)
                        profit_m = np.tanh(fund_row.get('profit_margin', 0) / 50)

                        stock_features.extend([roe, roa, debt_eq, curr_ratio, pe_ratio, pb_ratio, profit_m])
                    except KeyError:
                        # Fundamental data yok, 0 ile doldur
                        stock_features.extend([0] * 7)

                features.extend(stock_features)

            except KeyError:
                # Missing data: forward-fill from last known step, then warn (#4)
                logger.warning(f"Missing data for {symbol} on {current_date} — "
                               f"filling with last known observation features.")
                n_features = 10 if self.phase == 1 else 17
                if len(features) >= n_features:
                    # repeat last symbol's feature block
                    features.extend(features[-n_features:])
                else:
                    features.extend([0.0] * n_features)

        # State vector oluştur
        state_components = [[balance_norm], shares_norm, features]

        # Prediction features: 4 per stock
        if self.use_predictions and self.prediction_features is not None:
            pred_features = []
            current_date_str = str(current_date)[:10]
            date_preds = self.prediction_features.get(current_date_str, {})

            for symbol in self.symbols:
                sym_pred = date_preds.get(symbol, {})
                pred_features.extend([
                    np.tanh(sym_pred.get('predicted_return', 0.0) * 10),
                    sym_pred.get('predicted_direction', 0.0) * 2 - 1,
                    sym_pred.get('confidence', 0.5) * 2 - 1,
                    sym_pred.get('ensemble_agreement', 0.5) * 2 - 1,
                ])

            state_components.append(pred_features)

        # Faz 2: Macro features ekle (tüm stocks için paylaşılan)
        if self.phase == 2 and self.macro_df is not None:
            try:
                # Timezone-aware comparison için current_date'i normalize et
                current_date_norm = pd.Timestamp(current_date).tz_localize(None)

                # Macro data için current date'e en yakın tarihi bul (forward fill)
                macro_dates = self.macro_df.index[self.macro_df.index <= current_date_norm]
                if len(macro_dates) > 0:
                    macro_date = macro_dates[-1]
                    macro_row = self.macro_df.loc[macro_date]

                    # 6 macro indicators (normalized)
                    policy_rate = np.tanh(macro_row.get('policy_rate', 0) / 50)
                    cpi = np.tanh(macro_row.get('cpi_inflation', 0) / 100)
                    ppi = np.tanh(macro_row.get('ppi_inflation', 0) / 100)
                    usd = np.tanh(macro_row.get('usd_try', 30) / 50)
                    eur = np.tanh(macro_row.get('eur_try', 35) / 50)
                    bist100 = np.tanh(macro_row.get('bist100_index', 5000) / 10000)

                    macro_features = [policy_rate, cpi, ppi, usd, eur, bist100]
                else:
                    # Tarih çok erken, 0 kullan
                    macro_features = [0] * 6

                state_components.append(macro_features)
            except Exception as e:
                logger.warning(f"Macro data error at {current_date}: {e}")
                state_components.append([0] * 6)

        # Final state vector — clip to observation space bounds (#1)
        state = np.concatenate(state_components).astype(np.float32)
        state = np.clip(state, -10.0, 10.0)

        return state

    def _get_atr(self, symbol: str) -> float:
        """Sembol icin ATR_N hesapla (son atr_period gun).

        Veri yetersizse 0.0 dondurur; caller None gibi yorumlamalı.
        """
        current_date = self._get_current_date()
        try:
            sym_df = self.df.xs(symbol, level='symbol')
            sym_df = sym_df[sym_df.index <= current_date].tail(self.atr_period + 1)
            if len(sym_df) < 2:
                return 0.0

            high = sym_df['high'].astype(float).values
            low = sym_df['low'].astype(float).values
            close = sym_df['close'].astype(float).values

            # True Range: max(H-L, |H-Cprev|, |L-Cprev|)
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]),
                )
            )
            return float(np.mean(tr))
        except Exception:
            return 0.0

    def _atr_position_size(self, symbol: str, action_signal: float) -> int:
        """ATR-tabanlı pozisyon boyutu hesapla.

        Args:
            symbol: Hisse sembolü
            action_signal: Ham model çıktısı [-1, +1]

        Returns:
            İşlem yapılacak lot sayısı (pozitif=alım, negatif=satış)
        """
        current_price = self._get_current_price(symbol)
        if current_price <= 0:
            return 0

        atr = self._get_atr(symbol)
        atr_stop = atr * self.atr_multiplier if atr > 0 else 0.0

        portfolio_value = self._get_portfolio_value()
        position_value = portfolio_value * self.risk_per_trade * abs(action_signal)

        if atr_stop > 0:
            shares = position_value / (current_price * atr_stop / current_price)
        else:
            # ATR hesaplanamadıysa sabit lot'a düş
            shares = abs(action_signal) * self.max_shares_per_trade

        # Maksimum pozisyon sınırı: portfolio'nun max_position_pct'i
        max_shares_by_pct = (portfolio_value * self.max_position_pct) / current_price
        shares = min(shares, max_shares_by_pct)

        # Mutlak üst sınır
        shares = min(shares, self.max_shares_per_trade * 10)

        shares_int = max(1, int(round(shares))) if shares >= 0.5 else 0
        return int(np.sign(action_signal)) * shares_int

    def _kelly_position_size(self, symbol: str, action_signal: float, win_prob: float) -> int:
        """Kelly Kriteri ile pozisyon boyutu hesapla.

        f* = (p*b - q*a) / (a*b)
        position = f* * kelly_fraction

        Args:
            symbol: Hisse sembolü
            action_signal: Ham model çıktısı [-1, +1]
            win_prob: Kazanma olasılığı (direction confidence, 0-1)

        Returns:
            İşlem yapılacak lot sayısı
        """
        if win_prob <= 0 or win_prob >= 1:
            # Geçersiz olasılık — sabit lot'a düş
            return int(np.sign(action_signal)) * max(1, int(abs(action_signal) * self.max_shares_per_trade))

        current_price = self._get_current_price(symbol)
        if current_price <= 0:
            return 0

        p = win_prob
        q = 1.0 - p
        # Basitleştirilmiş Kelly: ort. kazanç/kayıp oranı ATR'den türet
        atr = self._get_atr(symbol)
        if atr > 0 and current_price > 0:
            b = atr / current_price   # ort. kazanç (ATR oranı)
            a = atr / current_price   # ort. kayıp (simetrik varsayım)
        else:
            b = 0.05   # %5 ort. kazanç (varsayılan)
            a = 0.05

        if a <= 0 or b <= 0:
            return 0

        kelly_f = (p * b - q * a) / (a * b)
        kelly_f = max(0.0, kelly_f)  # negatif Kelly = işlem yapma

        position_fraction = kelly_f * self.kelly_fraction * abs(action_signal)

        portfolio_value = self._get_portfolio_value()
        position_value = portfolio_value * position_fraction
        max_value = portfolio_value * self.max_position_pct

        position_value = min(position_value, max_value)
        shares = position_value / current_price if current_price > 0 else 0.0
        shares = min(shares, self.max_shares_per_trade * 10)

        shares_int = max(1, int(round(shares))) if shares >= 0.5 else 0
        return int(np.sign(action_signal)) * shares_int

    def _get_current_price(self, symbol: str) -> float:
        """Hissenin güncel fiyatını döndür (close price)"""
        current_date = self._get_current_date()
        try:
            price = self.df.loc[(symbol, current_date), 'close']
            # Convert to float, handling various pandas types
            if isinstance(price, (int, float, np.integer, np.floating)):
                return float(price)
            else:
                # Try conversion, this handles most pandas scalar types
                try:
                    return float(price)  # type: ignore
                except:
                    logger.warning(f"Could not convert price to float: {type(price)}")
                    return 0.0
        except KeyError:
            logger.warning(f"Price not found for {symbol} on {current_date}")
            return 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Price conversion error for {symbol} on {current_date}: {e}")
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
