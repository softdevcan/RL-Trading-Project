"""
Base Hyperparameter Optimizer

Tüm algoritma optimizerları için temel sınıf.
Optuna entegrasyonu, pruning, callbacks ve evaluation logic içerir.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
from env.trading_env import TradingEnv
from data.data_fetcher import DataFetcher
from hyperparameter_optimization.search_spaces import get_search_space, suggest_hyperparameter, SearchSpaceConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """
    Training progress gösterir - her episode'da güncellenir.
    """

    def __init__(self, trial_number: int, total_timesteps: int):
        super().__init__()
        self.trial_number = trial_number
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.episode_count = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        logger.info(f"Trial {self.trial_number}: Training started (0/{self.total_timesteps} steps, Episode 0)")

    def _on_step(self) -> bool:
        # Her episode bittiğinde progress göster
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_count += 1

                elapsed = time.time() - self.start_time
                progress = (self.n_calls / self.total_timesteps) * 100
                steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
                eta = (self.total_timesteps - self.n_calls) / steps_per_sec if steps_per_sec > 0 else 0

                # Episode reward
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                logger.info(
                    f"Trial {self.trial_number} | Episode {self.episode_count}: "
                    f"Step {self.n_calls}/{self.total_timesteps} ({progress:.1f}%) | "
                    f"Reward: {episode_reward:.2f} | Length: {episode_length} | "
                    f"{steps_per_sec:.0f} steps/s | ETA: {eta:.0f}s"
                )
        return True


class TrialPruningCallback(BaseCallback):
    """
    Optuna pruning callback.
    Training sırasında kötü performans gösteren trial'ları erken durdurur.
    """

    def __init__(self, trial: optuna.Trial, eval_freq: int, pruning_start_step: int = 0):
        super().__init__()
        self.trial = trial
        self.eval_freq = eval_freq
        self.pruning_start_step = pruning_start_step
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls >= self.pruning_start_step:
            # Get current reward from training
            if len(self.locals.get("infos", [])) > 0:
                info = self.locals["infos"][0]
                if "episode" in info:
                    mean_reward = info["episode"]["r"]

                    # Report to Optuna
                    self.trial.report(mean_reward, self.eval_idx)

                    # Check if should prune
                    if self.trial.should_prune():
                        self.is_pruned = True
                        logger.info(f"Trial {self.trial.number} pruned at step {self.n_calls}")
                        return False

                    self.eval_idx += 1

        return True


class BaseHyperparameterOptimizer:
    """
    Temel hiper parametre optimizer sınıfı.
    Tüm algoritma-specific optimizer'lar bu sınıftan türer.
    """

    def __init__(
        self,
        algorithm_name: str,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        seed: Optional[int] = None,
        phase: int = 2,
        reward_type: str = 'psr'
    ):
        """
        Args:
            algorithm_name: Algoritma ismi ('ppo', 'a2c', 'td3', 'sac')
            study_name: Optuna study ismi (None ise otomatik oluşturulur)
            storage: Optuna storage (None ise in-memory, önerilir: 'sqlite:///results/hyperparameter_studies/optuna_studies.db')
            n_trials: Toplam deneme sayısı
            n_jobs: Paralel çalıştırma sayısı (1 = sequential)
            sampler: Optuna sampler (None ise TPESampler)
            pruner: Optuna pruner (None ise MedianPruner)
            seed: Random seed
            phase: 1=Faz1 (56 features), 2=Faz2 (97 features)
            reward_type: 'simple' or 'psr'
        """
        self.algorithm_name = algorithm_name.lower()
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.phase = phase
        self.reward_type = reward_type

        # Study name
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"{self.algorithm_name}_optimization_{timestamp}"
        self.study_name = study_name

        # Storage
        if storage is None:
            # Anchor to repo root so the path is independent of the caller's CWD
            # (the FastAPI server is launched from various scripts/IDEs). Use an
            # absolute path with forward slashes so the SQLAlchemy URI parses
            # correctly on Windows — `sqlite:///C:\a\b.db` mishandles backslashes.
            project_root = Path(__file__).resolve().parent.parent
            db_path = project_root / "results" / "hyperparameter_studies" / "optuna_studies.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{db_path.as_posix()}"
        self.storage = storage

        # Sampler (Bayesian Optimization)
        if sampler is None:
            # Don't use seed for better exploration across different algorithms
            if seed is None:
                sampler = TPESampler(multivariate=True, warn_independent_sampling=False)
            else:
                sampler = TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False)
        self.sampler = sampler

        # Pruner (Early stopping)
        if pruner is None:
            # Adjusted for smaller n_trials: start pruning after first 20% of trials
            # and warmup for ~20% of total_timesteps (assuming 50k-100k timesteps)
            pruner = MedianPruner(
                n_startup_trials=max(2, n_trials // 5),  # 20% of trials or min 2
                n_warmup_steps=10000  # ~20% of 50k timesteps
            )
        self.pruner = pruner

        # Search space
        self.search_space = get_search_space(self.algorithm_name)

        # Results
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.study: Optional[optuna.Study] = None

        logger.info(f"✅ {self.algorithm_name.upper()} Optimizer initialized")
        logger.info(f"   Study: {self.study_name}")
        logger.info(f"   Trials: {self.n_trials}, Jobs: {self.n_jobs}")

    def create_study(self) -> optuna.Study:
        """Optuna study oluşturur veya yükler."""
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction="maximize",  # Sharpe ratio veya cumulative return maximize edilir
            load_if_exists=True,
        )
        return study

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Trial için hiper parametreler önerir.

        Args:
            trial: Optuna trial

        Returns:
            Önerilen hiper parametreler
        """
        params = {}
        for param_name, param_config in self.search_space.items():
            params[param_name] = suggest_hyperparameter(trial, param_name, param_config)

        return params

    def create_env(
        self,
        stock_symbols: list,
        start_date: str,
        end_date: str,
        phase: int = 2,
        reward_type: str = 'psr',
        use_cached_data: bool = True,
    ) -> gym.Env:
        """
        Trading environment oluşturur.

        Args:
            stock_symbols: Hisse senedi sembolleri
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            phase: 1=Faz1 (56 features), 2=Faz2 (97 features with fundamental+macro)
            reward_type: 'simple' (baseline) or 'psr' (risk-aware)
            use_cached_data: True ise data/bist/raw_stock_data.csv'den okur (CSV
                ezilmez, yfinance çağrısı yok). Dosya yoksa fallback yfinance.

        Returns:
            Trading environment
        """
        # 1. Fetch market data
        df = None
        data_fetcher = None  # asagidaki temizlik adimi bunu her dalda bekliyor
        if use_cached_data:
            try:
                data_fetcher = DataFetcher(start_date=start_date, end_date=end_date)
                full_df = data_fetcher.load_data('raw_stock_data.csv')

                # The CSV stores ALL cached symbols/dates — filter to the requested
                # symbols and date window. Match the user-supplied date strings to
                # the index's tz (naive vs aware) to avoid an incompatible-compare
                # error when the CSV was saved from a tz-aware yfinance pull.
                full_df = full_df[full_df.index.get_level_values('symbol').isin(stock_symbols)]
                idx_dates = full_df.index.get_level_values('date')
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                if idx_dates.tz is not None:
                    start_ts = start_ts.tz_localize(idx_dates.tz)
                    end_ts = end_ts.tz_localize(idx_dates.tz)
                mask = (idx_dates >= start_ts) & (idx_dates <= end_ts)
                df = full_df[mask]
                missing = set(stock_symbols) - set(df.index.get_level_values('symbol').unique())
                if missing:
                    logger.warning(
                        f"Cached CSV missing symbols {missing}; falling back to yfinance"
                    )
                    df = None
                elif df.empty:
                    logger.warning(
                        f"Cached CSV has no rows for {start_date}..{end_date}; "
                        f"falling back to yfinance"
                    )
                    df = None
                else:
                    logger.info(
                        f"📂 Loaded cached data: {len(df)} rows, "
                        f"{df.index.get_level_values('date').min().date()} to "
                        f"{df.index.get_level_values('date').max().date()}"
                    )
            except FileNotFoundError:
                logger.warning("Cached CSV not found; falling back to yfinance")
                df = None

        if df is None:
            data_fetcher = DataFetcher(start_date=start_date, end_date=end_date)
            df = data_fetcher.fetch_stock_data(stock_symbols)

        # Temizlik ZORUNLU — bu yol 'raw_stock_data.csv'yi ham yukluyordu.
        # Ham dosyada negatif fiyatlar var (yfinance'in 2005 TL sadelestirmesi
        # oncesine ait duzeltme artefaktlari). Negatif fiyat TradingEnv'de
        # bakiyeyi -initial_balance'in altina cekiyor, gozlemdeki log() NaN
        # uretiyor ve trial "Normal(loc: nan)" ile cokuyordu.
        if df is not None and not df.empty:
            if data_fetcher is None:
                data_fetcher = DataFetcher(start_date=start_date, end_date=end_date)
            rows_before = len(df)
            df = data_fetcher.clean_data(df)
            if len(df) != rows_before:
                logger.info(
                    f"🧹 Temizlik: {rows_before - len(df):,} satir dusuruldu "
                    f"(negatif/sifir fiyat, duplicate) -> {len(df):,} satir kaldi"
                )

        if df is None or df.empty:
            raise ValueError(f"No data fetched for {stock_symbols}")

        # 2. Load Phase 2 data (fundamental + macro) if needed
        fundamental_df = None
        macro_df = None

        if phase == 2:
            from data.fundamental_fetcher import FundamentalDataFetcher
            from data.macro_fetcher import MacroDataFetcher

            # Load fundamental data
            try:
                fund_fetcher = FundamentalDataFetcher()
                fundamental_df = fund_fetcher.load_data('fundamental_data.csv')
                logger.info(f"✅ Loaded fundamental data: {len(fundamental_df)} symbols")
            except FileNotFoundError:
                logger.warning("⚠️  Fundamental data not found, fetching...")
                fundamental_df = fund_fetcher.fetch_fundamental_data(stock_symbols, save=True)

            # Load macro data
            try:
                macro_fetcher = MacroDataFetcher(api_key="tV4qq6RzPr")
                macro_df = macro_fetcher.load_data('macro_data.csv')
                logger.info(f"✅ Loaded macro data: {len(macro_df)} rows")
            except FileNotFoundError:
                logger.warning("⚠️  Macro data not found, fetching...")
                macro_df = macro_fetcher.fetch_macro_data(save=True)

        # 3. Create environment
        env = TradingEnv(
            df=df,
            initial_balance=1_000_000,
            commission_rate=0.001,
            max_shares_per_trade=100,
            phase=phase,
            reward_type=reward_type,
            fundamental_df=fundamental_df,
            macro_df=macro_df
        )

        # 4. Monitor for statistics
        log_dir = f"logs/hyperopt_{self.algorithm_name}_phase{phase}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)

        logger.info(f"Environment created: Phase {phase}, Reward: {reward_type.upper()}")
        return env

    def create_model(self, trial: optuna.Trial, env: gym.Env, hyperparams: Dict[str, Any]):
        """
        Model oluşturur. Alt sınıflar tarafından override edilmeli.

        Args:
            trial: Optuna trial
            env: Gymnasium environment
            hyperparams: Hiper parametreler

        Returns:
            Stable-Baselines3 model instance
        """
        raise NotImplementedError("Alt sınıflar create_model metodunu implement etmeli")

    def evaluate_model(self, model, env: gym.Env, n_eval_episodes: int = 5) -> Dict[str, float]:
        """
        Model performansını değerlendirir.

        Args:
            model: Trained model
            env: Evaluation environment
            n_eval_episodes: Episode sayısı

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_sharpe_ratios = []
        episode_returns = []

        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            final_info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # Episode bittiğinde info'yu kaydet
                if done:
                    final_info = info

            episode_rewards.append(episode_reward)

            # Final info'dan metrics al (episode bittiğinde ekleniyor)
            if 'sharpe_ratio' in final_info:
                episode_sharpe_ratios.append(final_info['sharpe_ratio'])
            if 'cumulative_return' in final_info:
                episode_returns.append(final_info['cumulative_return'])

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_sharpe_ratio': np.mean(episode_sharpe_ratios) if episode_sharpe_ratios else 0.0,
            'mean_return': np.mean(episode_returns) if episode_returns else 0.0,
        }

        return metrics

    def objective(
        self,
        trial: optuna.Trial,
        stock_symbols: list,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        total_timesteps: int = 100_000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        use_cached_data: bool = True,
    ) -> float:
        """
        Optuna objective function.

        Args:
            trial: Optuna trial
            stock_symbols: Stock symbols
            train_start: Training start date
            train_end: Training end date
            val_start: Validation start date
            val_end: Validation end date
            total_timesteps: Training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes

        Returns:
            Objective value (Sharpe ratio veya cumulative return)
        """
        try:
            # Suggest hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)

            logger.info(f"\n{'='*80}")
            logger.info(f"Trial {trial.number}: Testing hyperparameters")
            logger.info(f"{'='*80}")
            for key, value in hyperparams.items():
                logger.info(f"  {key}: {value}")

            # Create environments with Phase and Reward Type
            train_env = self.create_env(stock_symbols, train_start, train_end,
                                       phase=self.phase, reward_type=self.reward_type,
                                       use_cached_data=use_cached_data)
            val_env = self.create_env(stock_symbols, val_start, val_end,
                                     phase=self.phase, reward_type=self.reward_type,
                                     use_cached_data=use_cached_data)

            # Create model
            model = self.create_model(trial, train_env, hyperparams)

            # Callbacks
            progress_callback = ProgressCallback(trial.number, total_timesteps)
            pruning_callback = TrialPruningCallback(trial, eval_freq=eval_freq)
            callbacks = [progress_callback, pruning_callback]

            # Train
            start_time = time.time()
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
            train_time = time.time() - start_time

            # Check if pruned
            if pruning_callback.is_pruned:
                raise optuna.TrialPruned()

            # Evaluate on validation set
            metrics = self.evaluate_model(model, val_env, n_eval_episodes)

            # Log results with timing breakdown
            total_time = time.time() - start_time
            eval_time = total_time - train_time

            logger.info(f"\n{'='*80}")
            logger.info(f"Trial {trial.number} Results:")
            logger.info(f"  Mean Reward: {metrics['mean_reward']:.4f}")
            logger.info(f"  Sharpe Ratio: {metrics['mean_sharpe_ratio']:.4f}")
            logger.info(f"  Cumulative Return: {metrics['mean_return']:.4f}")
            logger.info(f"  Training Time: {train_time:.2f}s")
            logger.info(f"  Evaluation Time: {eval_time:.2f}s")
            logger.info(f"  Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
            logger.info(f"{'='*80}\n")

            # Store additional metrics
            trial.set_user_attr("mean_reward", metrics['mean_reward'])
            trial.set_user_attr("sharpe_ratio", metrics['mean_sharpe_ratio'])
            trial.set_user_attr("cumulative_return", metrics['mean_return'])
            trial.set_user_attr("training_time", train_time)
            trial.set_user_attr("evaluation_time", eval_time)
            trial.set_user_attr("total_time", total_time)

            # Return objective (maximize Sharpe ratio)
            return metrics['mean_sharpe_ratio']

        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} was pruned")
            raise

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            raise

        finally:
            # Cleanup
            if 'train_env' in locals():
                train_env.close()
            if 'val_env' in locals():
                val_env.close()

    def optimize(
        self,
        stock_symbols: list,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        total_timesteps: int = 100_000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        show_progress_bar: bool = True,
        use_cached_data: bool = True,
    ) -> optuna.Study:
        """
        Hiper parametre optimizasyonu çalıştırır.

        Args:
            stock_symbols: Stock symbols (e.g., ['THYAO.IS', 'SAHOL.IS'])
            train_start: Training start date (e.g., '2018-01-01')
            train_end: Training end date
            val_start: Validation start date
            val_end: Validation end date
            total_timesteps: Training timesteps per trial
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            show_progress_bar: Show Optuna progress bar

        Returns:
            Optimized Optuna study
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting Hyperparameter Optimization for {self.algorithm_name.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"Stocks: {stock_symbols}")
        logger.info(f"Train Period: {train_start} to {train_end}")
        logger.info(f"Val Period: {val_start} to {val_end}")
        logger.info(f"Trials: {self.n_trials}, Jobs: {self.n_jobs}")
        logger.info(f"{'='*80}\n")

        # Create study
        self.study = self.create_study()

        # Track total optimization time
        optimization_start = time.time()

        # Optimize
        self.study.optimize(
            lambda trial: self.objective(
                trial,
                stock_symbols,
                train_start,
                train_end,
                val_start,
                val_end,
                total_timesteps,
                eval_freq,
                n_eval_episodes,
                use_cached_data,
            ),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress_bar,
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Calculate timing statistics
        total_optimization_time = time.time() - optimization_start
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if completed_trials:
            trial_times = [t.user_attrs.get('total_time', 0) for t in completed_trials]
            avg_trial_time = sum(trial_times) / len(trial_times) if trial_times else 0
            min_trial_time = min(trial_times) if trial_times else 0
            max_trial_time = max(trial_times) if trial_times else 0
        else:
            avg_trial_time = min_trial_time = max_trial_time = 0

        # Store timing stats for saving
        self.timing_stats = {
            'total_optimization_time_seconds': total_optimization_time,
            'total_optimization_time_minutes': total_optimization_time / 60,
            'completed_trials': len(completed_trials),
            'total_trials': self.n_trials,
            'average_trial_time_seconds': avg_trial_time,
            'fastest_trial_seconds': min_trial_time,
            'slowest_trial_seconds': max_trial_time,
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Optimization Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Best Sharpe Ratio: {self.best_value:.4f}")
        logger.info(f"Best Hyperparameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\n⏱️  Timing Statistics:")
        logger.info(f"  Total Optimization Time: {total_optimization_time/60:.2f} minutes ({total_optimization_time:.1f}s)")
        logger.info(f"  Completed Trials: {len(completed_trials)}/{self.n_trials}")
        logger.info(f"  Average Trial Time: {avg_trial_time:.2f}s ({avg_trial_time/60:.2f} min)")
        logger.info(f"  Fastest Trial: {min_trial_time:.2f}s")
        logger.info(f"  Slowest Trial: {max_trial_time:.2f}s")
        logger.info(f"  Time per Trial Range: {min_trial_time:.1f}s - {max_trial_time:.1f}s")
        logger.info(f"{'='*80}\n")

        return self.study

    def save_best_params(self, filepath: Optional[str] = None):
        """En iyi parametreleri JSON olarak kaydeder."""
        if self.best_params is None:
            logger.warning("No best parameters to save. Run optimization first.")
            return

        if filepath is None:
            filepath = f"results/hyperparameter_studies/best_params_{self.algorithm_name}_{self.study_name}.json"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Collect all trial details
        trial_details = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_details.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'training_time_seconds': trial.user_attrs.get('training_time', 0),
                    'evaluation_time_seconds': trial.user_attrs.get('evaluation_time', 0),
                    'total_time_seconds': trial.user_attrs.get('total_time', 0),
                    'sharpe_ratio': trial.user_attrs.get('sharpe_ratio', 0),
                    'mean_reward': trial.user_attrs.get('mean_reward', 0),
                    'cumulative_return': trial.user_attrs.get('cumulative_return', 0),
                })

        result_data = {
            'algorithm': self.algorithm_name,
            'study_name': self.study_name,
            'best_value': self.best_value,
            'best_params': self.best_params,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().isoformat(),
            'timing_statistics': self.timing_stats if hasattr(self, 'timing_stats') else {},
            'all_trials': trial_details,
        }

        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"✅ Best parameters saved to {filepath}")
        logger.info(f"   Saved {len(trial_details)} trial details with timing information")
