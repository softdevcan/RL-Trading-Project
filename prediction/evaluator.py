"""
Gelismis Tahmin Degerlendirici

Gercek fiyatlarla tahminleri karsilastirip kapsamli metrik uretir.

Metrikler:
- MAPE, RMSE, MAE (fiyat dogrulugu)
- Direction Accuracy (yon dogrulugu — trading icin en kritik)
- Profit Factor (dogru yon getirisi / yanlis yon kaybi)
- Information Coefficient (tahmin-gercek rank korelasyonu)
- Calibration (guven skoru ile gercek dogruluk uyumu)
- Regime-based evaluation (bull/bear/sideways ayri performans)
- Diebold-Mariano testi (model karsilastirma)
- Backtest simulasyonu (tahmine dayali strateji P&L)
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class PredictionEvaluator:
    """Kapsamli tahmin performans degerlendirici."""

    # ------------------------------------------------------------------
    # Temel Metrikler
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_batch(
        predictions: List[Dict[str, Any]],
        actuals: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Tahmin listesini gercek kapanis fiyatlariyla degerlendir."""
        results = []
        for pred in predictions:
            symbol = pred['symbol']
            actual = actuals.get(symbol)
            if actual is None:
                results.append({**pred, 'actual_close': None,
                                'error_pct': None, 'direction_correct': None})
                continue

            predicted = pred['predicted_close']
            error_pct = (predicted - actual) / actual * 100 if actual != 0 else None

            current = pred.get('current_close', actual)
            pred_dir = predicted > current
            actual_dir = actual > current
            direction_correct = pred_dir == actual_dir

            results.append({
                **pred,
                'actual_close': round(float(actual), 4),
                'error_pct': round(float(error_pct), 2) if error_pct is not None else None,
                'direction_correct': direction_correct,
            })

        return results

    @staticmethod
    def compute_rolling_metrics(
        history: List[Dict[str, Any]], window: int = 30
    ) -> Dict[str, float]:
        """Son `window` tahmin uzerinden rolling metrik hesapla."""
        evaluated = [
            h for h in history
            if h.get('actual_close') is not None and h.get('error_pct') is not None
        ]

        if not evaluated:
            return {
                'mape': None, 'rmse': None, 'mae': None,
                'direction_accuracy': None, 'n_predictions': 0
            }

        recent = evaluated[-window:]

        errors = [abs(r['error_pct']) for r in recent if r['error_pct'] is not None]
        mape = float(np.mean(errors)) if errors else None

        pred_arr = np.array([r['predicted_close'] for r in recent])
        actual_arr = np.array([r['actual_close'] for r in recent])
        rmse = float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_arr - actual_arr)))

        dir_correct = [r['direction_correct'] for r in recent
                       if r.get('direction_correct') is not None]
        dir_acc = float(np.mean(dir_correct) * 100) if dir_correct else None

        return {
            'mape': round(mape, 4) if mape is not None else None,
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'direction_accuracy': round(dir_acc, 2) if dir_acc is not None else None,
            'n_predictions': len(recent),
        }

    # ------------------------------------------------------------------
    # Gelismis Metrikler
    # ------------------------------------------------------------------

    @staticmethod
    def compute_profit_factor(
        y_true: np.ndarray, y_pred: np.ndarray, current_prices: np.ndarray,
    ) -> float:
        """Profit Factor: dogru yon tahminlerinin getirisi / yanlis yon kaybi.

        > 1.0 = karli strateji, > 2.0 = iyi strateji
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        current_prices = np.asarray(current_prices)

        pred_direction = y_pred > current_prices
        actual_returns = (y_true - current_prices) / np.where(current_prices != 0, current_prices, 1)

        strategy_returns = np.where(pred_direction, actual_returns, -actual_returns)

        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))

        if gross_loss == 0:
            return 9999.99 if gross_profit > 0 else 0.0

        return round(float(gross_profit / gross_loss), 4)

    @staticmethod
    def compute_information_coefficient(
        y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Information Coefficient: tahmin ile gercek getiri arasi rank korelasyonu.

        IC > 0.05 iyi, > 0.10 cok iyi kabul edilir.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) < 5:
            return {'ic': 0.0, 'ic_pvalue': 1.0, 'rank_ic': 0.0}

        spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
        pearson_corr, pearson_p = stats.pearsonr(y_true, y_pred)

        return {
            'ic': round(float(pearson_corr), 4),
            'ic_pvalue': round(float(pearson_p), 6),
            'rank_ic': round(float(spearman_corr), 4),
            'rank_ic_pvalue': round(float(spearman_p), 6),
        }

    @staticmethod
    def compute_calibration(
        predictions: List[Dict[str, Any]],
        n_bins: int = 5,
    ) -> Dict[str, Any]:
        """Guven skoru kalibrasyonu: guven ile gercek dogruluk uyumu.

        Ideal: confidence=0.8 ise gercek dogruluk da ~0.8 olmali.
        """
        evaluated = [
            p for p in predictions
            if p.get('direction_correct') is not None and p.get('confidence') is not None
        ]

        if len(evaluated) < n_bins:
            return {'bins': [], 'ece': None}

        confidences = np.array([p['confidence'] for p in evaluated])
        accuracies = np.array([float(p['direction_correct']) for p in evaluated])

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        ece_sum = 0.0

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() == 0:
                continue

            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            count = int(mask.sum())

            bins.append({
                'confidence_range': [round(bin_edges[i], 2), round(bin_edges[i + 1], 2)],
                'avg_confidence': round(avg_conf, 4),
                'avg_accuracy': round(avg_acc, 4),
                'count': count,
            })

            ece_sum += abs(avg_acc - avg_conf) * count

        ece = ece_sum / len(evaluated) if evaluated else None

        return {
            'bins': bins,
            'ece': round(float(ece), 4) if ece is not None else None,
            'n_predictions': len(evaluated),
        }

    @staticmethod
    def compute_regime_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_prices: np.ndarray,
        volatility: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Farkli piyasa rejimlerinde ayri performans degerlendirme.

        Rejimler:
        - bull: fiyat yukarı trend
        - bear: fiyat asagi trend
        - high_vol: yuksek volatilite
        - low_vol: dusuk volatilite
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        current_prices = np.asarray(current_prices)

        actual_returns = (y_true - current_prices) / np.where(current_prices != 0, current_prices, 1)

        bull_mask = actual_returns > 0
        bear_mask = actual_returns < 0

        results = {}

        for regime_name, mask in [('bull', bull_mask), ('bear', bear_mask)]:
            if mask.sum() < 5:
                continue
            regime_metrics = {
                'n_samples': int(mask.sum()),
                'mape': round(float(np.mean(np.abs(
                    (y_true[mask] - y_pred[mask]) / np.where(y_true[mask] != 0, y_true[mask], 1)
                )) * 100), 4),
                'direction_accuracy': round(float(np.mean(
                    (y_pred[mask] > current_prices[mask]) == (y_true[mask] > current_prices[mask])
                ) * 100), 2),
            }
            results[regime_name] = regime_metrics

        if volatility is not None and len(volatility) == len(y_true):
            vol_median = np.median(volatility)
            high_vol = volatility > vol_median
            low_vol = ~high_vol

            for regime_name, mask in [('high_vol', high_vol), ('low_vol', low_vol)]:
                if mask.sum() < 5:
                    continue
                results[regime_name] = {
                    'n_samples': int(mask.sum()),
                    'mape': round(float(np.mean(np.abs(
                        (y_true[mask] - y_pred[mask]) / np.where(y_true[mask] != 0, y_true[mask], 1)
                    )) * 100), 4),
                    'direction_accuracy': round(float(np.mean(
                        (y_pred[mask] > current_prices[mask]) == (y_true[mask] > current_prices[mask])
                    ) * 100), 2),
                }

        return results

    @staticmethod
    def diebold_mariano_test(
        y_true: np.ndarray,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        model_1_name: str = 'Model A',
        model_2_name: str = 'Model B',
    ) -> Dict[str, Any]:
        """Diebold-Mariano testi: iki modelin tahmin hatasi istatistiksel olarak farkli mi?

        H0: iki modelin tahmin performansi esit
        H1: tahmin performanslari farkli

        p < 0.05 ise fark istatistiksel olarak anlamli.
        """
        y_true = np.asarray(y_true)
        pred_1 = np.asarray(pred_1)
        pred_2 = np.asarray(pred_2)

        e1 = (y_true - pred_1) ** 2
        e2 = (y_true - pred_2) ** 2
        d = e1 - e2

        n = len(d)
        if n < 10:
            return {
                'dm_statistic': None,
                'p_value': None,
                'significant': False,
                'better_model': None,
                'error': 'Yetersiz veri',
            }

        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)

        if d_var == 0:
            return {
                'dm_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'better_model': None,
            }

        dm_stat = d_mean / np.sqrt(d_var / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        better = model_1_name if d_mean < 0 else model_2_name

        return {
            'dm_statistic': round(float(dm_stat), 4),
            'p_value': round(float(p_value), 6),
            'significant': p_value < 0.05,
            'better_model': better if p_value < 0.05 else None,
            'model_1_name': model_1_name,
            'model_2_name': model_2_name,
            'model_1_mse': round(float(np.mean(e1)), 6),
            'model_2_mse': round(float(np.mean(e2)), 6),
        }

    @staticmethod
    def backtest_simulation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_prices: np.ndarray,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """Tahmine dayali basit long/short strateji backtest'i.

        Strateji: Tahmin yukaris ise long, asagi ise short (flat).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        current_prices = np.asarray(current_prices)

        capital = initial_capital
        positions = []
        equity_curve = [capital]

        for i in range(len(y_true)):
            pred_return = (y_pred[i] - current_prices[i]) / current_prices[i] if current_prices[i] != 0 else 0
            actual_return = (y_true[i] - current_prices[i]) / current_prices[i] if current_prices[i] != 0 else 0

            if pred_return > 0:
                pnl = capital * actual_return - capital * commission_rate * 2
                positions.append('LONG')
            else:
                pnl = -capital * commission_rate
                positions.append('FLAT')

            capital += pnl
            equity_curve.append(capital)

        total_return = (capital - initial_capital) / initial_capital * 100
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - peak) / np.where(peak != 0, peak, 1)
        max_drawdown = float(np.min(drawdowns)) * 100

        daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        n_long = positions.count('LONG')
        n_total = len(positions)
        win_count = sum(
            1 for i in range(len(y_true))
            if positions[i] == 'LONG' and y_true[i] > current_prices[i]
        )
        win_rate = win_count / n_long * 100 if n_long > 0 else 0.0

        return {
            'initial_capital': initial_capital,
            'final_capital': round(capital, 2),
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 4),
            'n_trades': n_long,
            'n_total': n_total,
            'win_rate': round(win_rate, 2),
            'equity_curve': [round(e, 2) for e in equity_curve[::max(1, len(equity_curve) // 100)]],
        }

    # ------------------------------------------------------------------
    # Chart Data
    # ------------------------------------------------------------------

    @staticmethod
    def build_prediction_chart_data(
        history: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Tahmin vs gercek chart verisi olustur."""
        evaluated = [
            h for h in history
            if h.get('actual_close') is not None
        ]
        evaluated = sorted(evaluated, key=lambda x: x.get('prediction_date', ''))

        return {
            'dates': [h['prediction_date'] for h in evaluated],
            'predicted': [h['predicted_close'] for h in evaluated],
            'actual': [h['actual_close'] for h in evaluated],
            'direction_correct': [h.get('direction_correct') for h in evaluated],
        }

    @staticmethod
    def build_accuracy_trend_data(
        history: List[Dict[str, Any]], window: int = 10
    ) -> Dict[str, List]:
        """Rolling MAPE ve yon dogruluk trendi olustur."""
        evaluated = [
            h for h in history
            if h.get('actual_close') is not None and h.get('error_pct') is not None
        ]
        evaluated = sorted(evaluated, key=lambda x: x.get('prediction_date', ''))

        dates, mapes, dir_accs = [], [], []

        for i in range(window, len(evaluated) + 1):
            batch = evaluated[i - window:i]
            errors = [abs(r['error_pct']) for r in batch if r['error_pct'] is not None]
            mape = float(np.mean(errors)) if errors else None
            dir_vals = [r['direction_correct'] for r in batch
                        if r.get('direction_correct') is not None]
            dir_acc = float(np.mean(dir_vals) * 100) if dir_vals else None

            dates.append(evaluated[i - 1]['prediction_date'])
            mapes.append(round(mape, 2) if mape is not None else None)
            dir_accs.append(round(dir_acc, 2) if dir_acc is not None else None)

        return {'dates': dates, 'mape': mapes, 'direction_accuracy': dir_accs}

    # ------------------------------------------------------------------
    # Gelismis Risk Metrikleri  (Faz 3.4.2)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Sortino Ratio: sadece asagi yonlu volatiliteyi cezalandirir.

        Sharpe'dan farklı olarak pozitif volatilite cezalandirilmaz.
        > 1.0 = kabul edilebilir, > 2.0 = iyi.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 2:
            return 0.0

        excess = r - risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        downside_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-9
        if downside_std < 1e-9:
            return 0.0
        return round(float(np.mean(excess) / downside_std * np.sqrt(periods_per_year)), 4)

    @staticmethod
    def compute_calmar_ratio(
        equity_curve: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Calmar Ratio: yillik getiri / maksimum drawdown.

        > 1.0 = kabul edilebilir, > 3.0 = iyi.
        """
        eq = np.asarray(equity_curve, dtype=float)
        if len(eq) < 2 or eq[0] <= 0:
            return 0.0

        total_return = (eq[-1] - eq[0]) / eq[0]
        annual_return = (1 + total_return) ** (periods_per_year / len(eq)) - 1

        peak = np.maximum.accumulate(eq)
        drawdowns = (eq - peak) / np.where(peak > 0, peak, 1)
        max_dd = abs(float(np.min(drawdowns)))

        if max_dd < 1e-9:
            return 0.0
        return round(float(annual_return / max_dd), 4)

    @staticmethod
    def compute_deflated_sharpe_ratio(
        returns: np.ndarray,
        n_trials: int = 1,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """Deflated Sharpe Ratio (DSR): coklu test ile sisman kuyruk etkisini duzeltir.

        Bailey & Lopez de Prado (2014) formulu.
        DSR > 0.95 ise strateji anlamli kabul edilir.

        Args:
            returns: Strateji getirileri
            n_trials: Test edilen strateji sayisi (daha fazla = daha siki duzeltme)
        """
        from scipy.stats import norm as sp_norm

        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 10:
            return {'dsr': 0.0, 'psr': 0.0, 'sr': 0.0}

        sr = float(np.mean(r) / (np.std(r, ddof=1) + 1e-9) * np.sqrt(periods_per_year))

        skew = float(pd.Series(r).skew())
        kurt = float(pd.Series(r).kurtosis())  # excess kurtosis

        # SR* benchmark (en iyi beklenen SR n_trials denemeden)
        gamma = 0.5772156649  # Euler-Mascheroni sabiti
        sr_star = (1 - gamma) * sp_norm.ppf(1 - 1 / n_trials) + gamma * sp_norm.ppf(
            1 - 1 / (n_trials * np.e)
        ) if n_trials > 1 else 0.0

        # PSR: SR'nin benchmark'i gecme olasiligi
        sigma_sr = np.sqrt((1 - skew * sr + (kurt / 4) * sr ** 2) / (n - 1))
        if sigma_sr < 1e-9:
            psr = 0.0
        else:
            psr = float(sp_norm.cdf((sr - sr_star) / sigma_sr))

        return {
            'sr': round(sr, 4),
            'sr_benchmark': round(sr_star, 4),
            'psr': round(psr, 4),
            'dsr': round(psr, 4),  # DSR ~ PSR bu formülasyonda
        }

    @staticmethod
    def compute_turnover(
        y_pred: np.ndarray,
        current_prices: np.ndarray,
        threshold: float = 0.0,
    ) -> Dict[str, float]:
        """Yon degisikliklerine dayali turnover analizi.

        Args:
            y_pred: Tahmin edilen fiyatlar
            current_prices: Guncel kapanış fiyatları
            threshold: Islem icin minimum tahmin degisimi (orani)

        Returns:
            turnover_rate: Yon degistiren gun orani
            avg_trade_size: Ortalama tahmin degisimi
            n_trades: Toplam islem sayisi
        """
        if len(y_pred) < 2:
            return {'turnover_rate': 0.0, 'avg_trade_size': 0.0, 'n_trades': 0}

        pred_returns = np.where(
            current_prices != 0,
            (y_pred - current_prices) / current_prices,
            0.0
        )
        positions = np.where(pred_returns > threshold, 1, np.where(pred_returns < -threshold, -1, 0))

        position_changes = np.diff(positions)
        n_trades = int(np.sum(position_changes != 0))
        turnover_rate = round(float(n_trades / max(len(positions) - 1, 1)), 4)
        avg_trade_size = round(float(np.mean(np.abs(pred_returns[np.abs(pred_returns) > threshold]))), 6) \
            if np.any(np.abs(pred_returns) > threshold) else 0.0

        return {
            'turnover_rate': turnover_rate,
            'n_trades': n_trades,
            'avg_trade_size': avg_trade_size,
        }

    @staticmethod
    def compute_comprehensive_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_prices: np.ndarray,
        volatility: Optional[np.ndarray] = None,
        n_trials: int = 1,
    ) -> Dict[str, Any]:
        """Tum metrikleri tek seferde hesapla (Faz 3.4.2 ile genisletildi)."""
        from prediction.models.base import BasePredictionModel

        basic = BasePredictionModel.compute_metrics(y_true, y_pred)
        ic = PredictionEvaluator.compute_information_coefficient(y_true, y_pred)
        profit_factor = PredictionEvaluator.compute_profit_factor(y_true, y_pred, current_prices)
        regime = PredictionEvaluator.compute_regime_metrics(y_true, y_pred, current_prices, volatility)
        backtest = PredictionEvaluator.backtest_simulation(y_true, y_pred, current_prices)

        # Faz 3.4.2: Gelismis risk metrikleri
        equity = np.array(backtest['equity_curve'], dtype=float)
        bt_returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1)

        sortino = PredictionEvaluator.compute_sortino_ratio(bt_returns)
        calmar = PredictionEvaluator.compute_calmar_ratio(equity)
        dsr = PredictionEvaluator.compute_deflated_sharpe_ratio(bt_returns, n_trials=n_trials)
        turnover = PredictionEvaluator.compute_turnover(y_pred, current_prices)

        return {
            'basic_metrics': basic,
            'information_coefficient': ic,
            'profit_factor': profit_factor,
            'regime_metrics': regime,
            'backtest': backtest,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'deflated_sharpe': dsr,
            'turnover': turnover,
        }
