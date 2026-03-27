"""
Reward Functions Module (Faz 2)
PSR (Portfolio-Sharpe-Returns) ve diğer risk-aware reward fonksiyonları

Referans: Ansari et al. (2024) - "A Multifaceted Approach to Stock Market Trading"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Faz 2 için gelişmiş reward hesaplayıcı

    Bileşenler:
    1. Portfolio Return (ana hedef)
    2. Differential Sharpe Ratio (risk-ayarlı getiri)
    3. MDD Penalty (aşağı yönlü risk)
    4. Volatility Penalty (toplam risk)
    5. Trade Frequency Bonus/Penalty (işlem kalitesi)
    """

    def __init__(
        self,
        w1: float = 0.50,  # Portfolio return weight
        w2: float = 0.30,  # Sharpe ratio weight
        w3: float = 0.10,  # MDD penalty weight
        w4: float = 0.05,  # Volatility penalty weight
        w5: float = 0.05,  # Trade frequency weight
        rolling_window: int = 30,      # Rolling metrics için window
        target_trades_per_100: int = 50,  # Hedef işlem sayısı (100 step başına)
        risk_free_rate: float = 0.15,     # Yıllık risksiz faiz (TCMB ~%15)
    ):
        """
        Args:
            w1-w5: Reward bileşenlerinin ağırlıkları
            rolling_window: Rolling Sharpe ve volatilite için window
            target_trades_per_100: 100 step başına hedeflenen işlem sayısı
            risk_free_rate: Yıllık risksiz faiz oranı (0.15 = %15)
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.rolling_window = rolling_window
        self.target_trades_per_100 = target_trades_per_100
        self.risk_free_rate = risk_free_rate

        # DSR (Differential Sharpe Ratio) için internal state
        self.dsr_a = 0.0  # Exponential moving average of returns
        self.dsr_b = 0.0  # Exponential moving average of squared returns
        self.dsr_eta = 1.0 / rolling_window  # Decay rate

        logger.info(f"RewardCalculator initialized with weights: "
                   f"Return={w1:.2f}, Sharpe={w2:.2f}, MDD={w3:.2f}, "
                   f"Vol={w4:.2f}, TradeFreq={w5:.2f}")

    def reset(self):
        """Episode başında DSR state'i sıfırla"""
        self.dsr_a = 0.0
        self.dsr_b = 0.0

    def calculate_psr_reward(
        self,
        portfolio_values: List[float],
        returns: List[float],
        current_step: int,
        trades_executed: int,
        commission_cost: float,
        initial_balance: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        PSR (Portfolio-Sharpe-Returns) reward hesapla

        Args:
            portfolio_values: Portfolio değerleri listesi (episode başından itibaren)
            returns: Günlük getiri oranları listesi
            current_step: Mevcut step
            trades_executed: Bu step'te yapılan işlem sayısı
            commission_cost: Bu step'teki komisyon maliyeti
            initial_balance: Başlangıç bakiyesi

        Returns:
            (total_reward, components_dict)
        """
        components = {}

        # 1. Portfolio Return Component
        if len(portfolio_values) < 2:
            portfolio_return = 0.0
        else:
            prev_value = portfolio_values[-2]
            curr_value = portfolio_values[-1]
            portfolio_return = ((curr_value - prev_value) / prev_value) * 100

        components['portfolio_return'] = portfolio_return

        # 2. Differential Sharpe Ratio (DSR)
        dsr = self._calculate_differential_sharpe_ratio(
            current_return=portfolio_return / 100  # Percentage to decimal
        )
        components['sharpe_ratio'] = dsr

        # 3. MDD Penalty
        mdd_penalty = self._calculate_mdd_penalty(portfolio_values)
        components['mdd_penalty'] = mdd_penalty

        # 4. Volatility Penalty
        volatility_penalty = self._calculate_volatility_penalty(returns)
        components['volatility_penalty'] = volatility_penalty

        # 5. Trade Frequency Bonus/Penalty
        trade_freq_component = self._calculate_trade_frequency_component(
            current_step=current_step,
            total_trades=trades_executed
        )
        components['trade_frequency'] = trade_freq_component

        # 6. Commission Penalty (basit)
        commission_penalty = (commission_cost / initial_balance) * 100
        components['commission'] = -commission_penalty

        # Total PSR Reward (weighted sum)
        total_reward = (
            self.w1 * portfolio_return +
            self.w2 * dsr +
            self.w3 * mdd_penalty +      # Already negative
            self.w4 * volatility_penalty +  # Already negative
            self.w5 * trade_freq_component -
            commission_penalty
        )

        components['total_reward'] = total_reward

        return total_reward, components

    def _calculate_differential_sharpe_ratio(self, current_return: float) -> float:
        """
        Differential Sharpe Ratio (DSR) hesapla

        DSR, Sharpe oranının bir time step'teki birinci dereceden katkısıdır.
        Online optimization için uygun, toplanabilir (additive) bir metriktir.

        Formula:
        A_t = A_{t-1} + eta * (R_t - A_{t-1})  # EMA of returns
        B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})  # EMA of squared returns
        DSR_t = (A_t - r_f) / sqrt(B_t - A_t^2 + epsilon)

        Args:
            current_return: Mevcut step'teki getiri (decimal, e.g., 0.01 = %1)

        Returns:
            DSR değeri (bounded to [-1, 1] with tanh)
        """
        # Daily risk-free rate (annual -> daily)
        daily_rf = self.risk_free_rate / 252

        # Update exponential moving averages
        self.dsr_a = self.dsr_a + self.dsr_eta * (current_return - self.dsr_a)
        self.dsr_b = self.dsr_b + self.dsr_eta * (current_return**2 - self.dsr_b)

        # Calculate DSR
        excess_return = self.dsr_a - daily_rf
        variance = self.dsr_b - self.dsr_a**2
        std_dev = np.sqrt(max(variance, 1e-9))  # Avoid division by zero

        dsr = excess_return / std_dev

        # Normalize DSR to [-1, 1] range for stability
        # Tipik Sharpe oranları -2 ile +4 arasında olur
        dsr_normalized = np.tanh(dsr / 2)  # Soft bound to [-1, 1]

        return dsr_normalized * 100  # Scale to match portfolio_return range

    def _calculate_mdd_penalty(self, portfolio_values: List[float]) -> float:
        """
        Maximum Drawdown (MDD) penalty hesapla

        MDD, portföyün en yüksek değerinden en düşük değerine olan en büyük düşüştür.
        Negatif değer olarak döner (ceza).

        Args:
            portfolio_values: Portfolio değerleri listesi

        Returns:
            MDD penalty (negative value, e.g., -5.0 for 5% drawdown)
        """
        if len(portfolio_values) < 2:
            return 0.0

        values = np.array(portfolio_values)
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        mdd = np.min(drawdown) * 100  # Percentage

        # MDD already negative, return as penalty
        # Küçük MDD'leri normalize et (örn: -1% çok az ceza, -20% çok ceza)
        penalty = mdd * abs(mdd) / 10  # Quadratic penalty for large drawdowns

        return penalty

    def _calculate_volatility_penalty(self, returns: List[float]) -> float:
        """
        Volatilite penalty hesapla

        Rolling window içinde volatilite (std dev) hesaplanır.
        Yüksek volatilite = yüksek risk = ceza.

        Args:
            returns: Günlük getiri oranları (percentage)

        Returns:
            Volatility penalty (negative value)
        """
        if len(returns) < 2:
            return 0.0

        # Rolling window kullan (yoksa tüm veriyi kullan)
        window_returns = returns[-self.rolling_window:]

        # Volatility (annualized)
        std_dev = np.std(window_returns)
        annualized_vol = std_dev * np.sqrt(252)

        # Normalize (BIST-30 tipik volatilite %20-40)
        # Hedef: %30 altı = düşük ceza, %30 üstü = yüksek ceza
        target_vol = 30.0
        excess_vol = max(0, annualized_vol - target_vol)

        # Penalty: quadratic (aşırı volatiliteyi daha çok cezalandır)
        penalty = -(excess_vol**2) / 100

        return penalty

    def _calculate_trade_frequency_component(
        self,
        current_step: int,
        total_trades: int
    ) -> float:
        """
        Trade frequency bonus/penalty hesapla

        Hedef: Aşırı işlemi (overtrading) ve aşırı muhafazakarlığı (undertrading) cezalandır.
        Optimal işlem aralığında bonus ver.

        Args:
            current_step: Mevcut step (episode içinde)
            total_trades: Toplam işlem sayısı

        Returns:
            Bonus/penalty değeri
        """
        if current_step == 0:
            return 0.0

        # 100 step başına normalize et
        trades_per_100 = (total_trades / current_step) * 100

        # Hedef aralık: 30-70 işlem / 100 step
        # (Ansari et al. başarılı modeller 50-60 civarında)
        lower_bound = self.target_trades_per_100 * 0.6  # 30
        upper_bound = self.target_trades_per_100 * 1.4  # 70

        if trades_per_100 < lower_bound:
            # Çok az işlem (aşırı muhafazakar)
            penalty = -(lower_bound - trades_per_100) / 10
            return penalty

        elif trades_per_100 > upper_bound:
            # Çok fazla işlem (overtrading)
            penalty = -(trades_per_100 - upper_bound) / 10
            return penalty

        else:
            # Optimal aralık, küçük bonus
            # En iyi noktada (50) maksimum bonus
            distance_from_optimal = abs(trades_per_100 - self.target_trades_per_100)
            bonus = max(0, (20 - distance_from_optimal) / 20) * 5  # Max +5 bonus

            return bonus

    def get_weights(self) -> Dict[str, float]:
        """Mevcut ağırlıkları döndür"""
        return {
            'w1_return': self.w1,
            'w2_sharpe': self.w2,
            'w3_mdd': self.w3,
            'w4_volatility': self.w4,
            'w5_trade_freq': self.w5
        }

    def set_weights(self, weights: Dict[str, float]):
        """Ağırlıkları güncelle (Optuna için)"""
        self.w1 = weights.get('w1_return', self.w1)
        self.w2 = weights.get('w2_sharpe', self.w2)
        self.w3 = weights.get('w3_mdd', self.w3)
        self.w4 = weights.get('w4_volatility', self.w4)
        self.w5 = weights.get('w5_trade_freq', self.w5)

        logger.info(f"Weights updated: {self.get_weights()}")


class SimpleRewardCalculator:
    """
    Faz 1 Basit Reward (Baseline için)

    Reward = Portfolio Change % - Commission Penalty
    """

    def __init__(self):
        logger.info("SimpleRewardCalculator initialized (Faz 1 baseline)")

    def reset(self):
        """Episode başında state sıfırla (basit reward için gerekli değil)"""
        pass

    def calculate_reward(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        commission_cost: float,
        initial_balance: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Basit reward hesapla

        Args:
            prev_portfolio_value: Önceki portfolio değeri
            current_portfolio_value: Mevcut portfolio değeri
            commission_cost: Komisyon maliyeti
            initial_balance: Başlangıç bakiyesi

        Returns:
            (reward, components_dict)
        """
        # Portfolio change percentage
        portfolio_change_pct = ((current_portfolio_value - prev_portfolio_value) /
                                prev_portfolio_value) * 100

        # Commission penalty
        commission_penalty = (commission_cost / initial_balance) * 100

        # Total reward
        reward = portfolio_change_pct - commission_penalty

        components = {
            'portfolio_change': portfolio_change_pct,
            'commission': -commission_penalty,
            'total_reward': reward
        }

        return reward, components


def test_reward_functions():
    """Test script for reward calculators"""
    print("\n" + "="*80)
    print("REWARD FUNCTIONS TEST")
    print("="*80)

    # Test 1: PSR Reward Calculator
    print("\n" + "-"*80)
    print("TEST 1: PSR Reward Calculator")
    print("-"*80)

    psr_calc = RewardCalculator(
        w1=0.50, w2=0.30, w3=0.10, w4=0.05, w5=0.05
    )

    # Simulate episode
    portfolio_values = [1_000_000]
    returns = []

    for step in range(10):
        # Simulate portfolio growth
        prev_value = portfolio_values[-1]
        change = np.random.randn() * 0.01  # ±1% random change
        new_value = prev_value * (1 + change)
        portfolio_values.append(new_value)
        returns.append(change * 100)

        # Calculate reward
        reward, components = psr_calc.calculate_psr_reward(
            portfolio_values=portfolio_values,
            returns=returns,
            current_step=step + 1,
            trades_executed=1,
            commission_cost=100,
            initial_balance=1_000_000
        )

        print(f"Step {step+1}: Reward={reward:+.4f}")
        print(f"  Components: Return={components['portfolio_return']:+.2f}, "
              f"Sharpe={components['sharpe_ratio']:+.2f}, "
              f"MDD={components['mdd_penalty']:+.2f}, "
              f"Vol={components['volatility_penalty']:+.2f}")

    # Test 2: Simple Reward Calculator
    print("\n" + "-"*80)
    print("TEST 2: Simple Reward Calculator (Baseline)")
    print("-"*80)

    simple_calc = SimpleRewardCalculator()

    prev_value = 1_000_000
    current_value = 1_010_000
    commission = 100

    reward, components = simple_calc.calculate_reward(
        prev_portfolio_value=prev_value,
        current_portfolio_value=current_value,
        commission_cost=commission,
        initial_balance=1_000_000
    )

    print(f"Reward: {reward:.4f}")
    print(f"Components: {components}")

    print("\n" + "="*80)
    print("[OK] REWARD FUNCTIONS TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_reward_functions()
