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


# Faz 9 / R1: tanimsiz dogrulama metrigi (model hicbir kontrol noktasinda islem
# yapmadi) icin amac degeri. NaN Optuna'da denemeyi BASARISIZ sayar; sonlu ve cok
# dusuk bir deger TPE'nin o bolgeden uzaklasmasini saglar.
UNDEFINED_OBJECTIVE = -1e6


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
            # Faz 9 / R1: rapor adimi KONTROL NOKTASI SIRASI (0..N_CHECKPOINTS-1) ve
            # raporlanan deger DOGRULAMA metrigi. Eskiden egitim bolum odulu, o da
            # yalniz `eval_freq` kati bolum sonuna denk gelirse raporlaniyor,
            # `n_warmup_steps=10000` ise rapor adimi 0,1,2.. oldugu icin budamayi
            # hic baslatmiyordu. Butcenin ilk yarisinda budama yok (tasarim varsayimi:
            # bir yillik dogrulamada erken kontrol noktalari gurultulu).
            from evaluation.rl_folds import N_CHECKPOINTS
            pruner = MedianPruner(
                n_startup_trials=max(2, n_trials // 5),  # 20% of trials or min 2
                n_warmup_steps=N_CHECKPOINTS // 2,
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

    def _phase2_frames(self, stock_symbols: list):
        """Faz 2 ortaminin temel + makro tablolari (rotayla ayni dosyalar)."""
        if self.phase != 2:
            return None, None
        from data.fundamental_fetcher import FundamentalDataFetcher
        from data.macro_fetcher import MacroDataFetcher

        fund_fetcher = FundamentalDataFetcher()
        try:
            fundamental_df = fund_fetcher.load_data('fundamental_data.csv')
        except FileNotFoundError:
            logger.warning("Fundamental data not found, fetching...")
            fundamental_df = fund_fetcher.fetch_fundamental_data(stock_symbols, save=True)
        macro_fetcher = MacroDataFetcher()  # anahtar .env'deki EVDS_API_KEY'den
        try:
            macro_df = macro_fetcher.load_data('macro_data.csv')
        except FileNotFoundError:
            logger.warning("Macro data not found, fetching...")
            macro_df = macro_fetcher.fetch_macro_data(save=True)
        return fundamental_df, macro_df

    def make_env(self, fold_data, role: str, *, train_env=None, monitor: bool = False) -> gym.Env:
        """Katin bir blogu icin ortam — egitim rotasiyla AYNI cagri (R1, Y21).

        Eskiden `create_env` ham `raw_stock_data.csv`'yi okuyordu: gosterge yok
        (10 ozelligin 5'i sabit varsayilan), varsayilan evren 5 sembol, pencereler
        istekten. Simdi panel, kat, olcek ve nedensel baglam
        `evaluation.rl_folds`'tan gelir; dogrulama ortami egitim ortami verilmeden
        kurulamaz (olcegin kaynagi; B9).
        """
        from evaluation import rl_folds
        env = rl_folds.make_fold_env(
            fold_data, role, train_env=train_env,
            price_norm=self.price_norm, price_norm_macd=self.price_norm_macd,
            **self.env_settings)
        if monitor:
            log_dir = f"logs/hyperopt_{self.algorithm_name}_phase{self.phase}"
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
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

    def _register_trial(self, trial, hyperparams, fold_data, total_timesteps, status,
                        metrics=None, returns=None, dates=None,
                        val_env=None, train_env=None, health=None,
                        selection_metric=None, selection=None):
        """Faz 9 / C2 — Optuna denemesini deneme kaydina yaz (kayit hatasi denemeyi bozmaz).

        Protokol rotayla ayni hesaptan, dogrulama ortaminin kendisinden ve
        olcegin alindigi EGITIM ortamindan olculur (`rl_eval_protocol`). Ortam
        kurulmadan biten ya da protokolu olculemeyen denemede maddeler None
        (beyan edilmemis = acik) — kayit yine yazilir, N'ye sayilir.
        """
        try:
            from evaluation.splits import is_canonical as _is_canonical
            from evaluation.trial_registry import CLEAN_PROTOCOL, register_trial, rl_eval_protocol
            try:
                # HPO denemesinin hiperparametreleri kendi yapilandirmasidir -> True
                protocol = rl_eval_protocol(val_env, train_env, health=health,
                                            hyperparams_clean=True,
                                            split_from_spec=_is_canonical(fold_data.spec),
                                            selection_metric=selection_metric)
            except Exception as exc:
                logger.warning(f"Deneme protokolu olculemedi ({exc}); maddeler acik yaziliyor")
                protocol = {k: None for k in CLEAN_PROTOCOL}
            ranges = fold_data.report['ranges']
            base_train = getattr(train_env, 'unwrapped', train_env) if train_env is not None else None
            settings = {k: v for k, v in self.env_settings.items()
                        if k not in ('fundamental_df', 'macro_df')}
            return register_trial(
                kind='rl_hpo',
                config={'algorithm': self.algorithm_name, 'phase': self.phase,
                        'reward_type': self.reward_type, 'hyperparameters': hyperparams,
                        'symbols': list(fold_data.symbols), 'train': ranges['train'],
                        'val': ranges['val'], 'total_timesteps': total_timesteps,
                        'missing_data_mode': 'own_last',
                        # rotanin kaydiyla ayni anahtarlar: soy kiyaslamasi bunlardan (R1)
                        'indicator_spec': self.indicator_spec,
                        'price_norm': base_train.get_price_norm() if base_train is not None else None,
                        **settings},
                seed=self.seed,
                status=status,
                metrics=metrics or {},
                returns=returns, dates=dates, split='val',
                fold=fold_data.fold,
                split_spec={'name': fold_data.spec.name, 'digest': fold_data.spec.digest()},
                protocol=protocol,
                extra={'study': getattr(trial.study, 'study_name', None),
                       'optuna_trial': trial.number,
                       'observation_health': health,
                       'val_range': ranges['val'],
                       'selection': selection},
            )
        except Exception as exc:
            logger.warning(f"Deneme kaydina yazilamadi ({exc})")
            return None

    def objective(self, trial: optuna.Trial, fold_data, total_timesteps: int = 100_000,
                  risk_free=None) -> float:
        """Optuna amac fonksiyonu (R1): egitim rotasinin YAPACAGI seyin testsiz hali.

        Katin egitim blogunda egit; butce boyunca esit aralikli kontrol
        noktalarinda DOGRULAMA blogunda risksiz faiz ustu Sharpe'i olc; amac =
        en iyi kontrol noktasinin metrigi (rotanin secim kuraliyla ayni). Test
        blogu burada HIC kurulmaz.
        """
        import shutil
        import tempfile

        from evaluation import rl_folds

        train_env = val_env = None
        hyperparams: Dict[str, Any] = {}
        ckpt_dir = tempfile.mkdtemp(prefix=f"hpo_{self.algorithm_name}_{trial.number}_")
        metric_name = (rl_folds.SELECTION_METRIC if risk_free is not None
                       else rl_folds.SELECTION_METRIC_RF0)
        try:
            hyperparams = self.suggest_hyperparameters(trial)
            logger.info(f"Trial {trial.number}: {hyperparams}")

            train_env = self.make_env(fold_data, 'train', monitor=True)
            # kayit ve protokol icin tek bir dogrulama ortami; olcumler taze ortamda
            val_env = self.make_env(fold_data, 'val', train_env=train_env)
            model = self.create_model(trial, train_env, dict(hyperparams))

            def on_checkpoint(idx: int, rec: dict) -> bool:
                value = rec['metric'] if rec['metric'] is not None else UNDEFINED_OBJECTIVE
                trial.report(value, idx)
                return not trial.should_prune()

            validation_cb = rl_folds.make_validation_callback(
                lambda: self.make_env(fold_data, 'val', train_env=train_env),
                total_timesteps=total_timesteps, checkpoint_dir=ckpt_dir,
                risk_free=risk_free, on_checkpoint=on_checkpoint)
            progress_callback = ProgressCallback(trial.number, total_timesteps)

            start_time = time.time()
            model.learn(total_timesteps=total_timesteps,
                        callback=[progress_callback, validation_cb])
            train_time = time.time() - start_time
            if validation_cb.stopped_early:
                raise optuna.TrialPruned()

            selection = rl_folds.select_checkpoint(validation_cb.records)
            selection_report = {
                'metric': metric_name, 'rule': 'argmax_validation',
                'selected_index': selection['index'], 'selected_step': selection['step'],
                'selected_metric': selection['metric'], 'degenerate': selection['degenerate'],
                'n_valid': selection['n_valid'], 'n_distinct': selection['n_distinct'],
                'n_negative_mean_excess': selection['n_negative_mean_excess'],
                'last_metric': selection['last_metric'],
                'pick_mean_excess_step': selection['pick_mean_excess_step'],
                'picks_agree': selection['picks_agree'],
                'checkpoints': [{k: v for k, v in r.items() if k != 'path'}
                                for r in validation_cb.records],
            }
            # secilen kontrol noktasinin dogrulama getirisi kayda girer (DSR'nin V'si)
            best_model = model if selection['final'] else type(model).load(
                selection['path'], device=model.device)
            episode = rl_folds.run_episode(best_model, val_env)
            value = selection['metric'] if not selection['degenerate'] else UNDEFINED_OBJECTIVE
            total_time = time.time() - start_time

            trial.set_user_attr("sharpe_ratio", selection['metric'])
            trial.set_user_attr("selection_metric", metric_name)
            trial.set_user_attr("selected_step", selection['step'])
            trial.set_user_attr("cumulative_return", selection.get('cumulative_return'))
            trial.set_user_attr("training_time", train_time)
            trial.set_user_attr("evaluation_time", total_time - train_time)
            trial.set_user_attr("total_time", total_time)

            # B9 bedeli (hakem-2 B9 #1, #8): dogrulama gozlemi egitim desteginde mi
            from evaluation.observation_health import observation_health
            health = observation_health(train_env, val_env)
            tid = self._register_trial(
                trial, hyperparams, fold_data, total_timesteps, 'completed',
                metrics={metric_name: selection['metric'],
                         'sharpe_rf0': selection.get('sharpe_rf0'),
                         'cumulative_return': selection.get('cumulative_return'),
                         'selected_step': selection['step']},
                returns=episode['returns'], dates=episode['dates'],
                val_env=val_env, train_env=train_env, health=health,
                selection_metric=metric_name, selection=selection_report)
            if tid:
                trial.set_user_attr("trial_registry_id", tid)
            logger.info(f"Trial {trial.number}: {metric_name}={selection['metric']} "
                        f"(adim {selection['step']}/{total_timesteps}, {total_time:.1f}s)")
            return float(value)

        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} was pruned")
            # Budanan deneme de bir denemedir: N'ye sayilir (hakem-2 #2a)
            self._register_trial(trial, hyperparams, fold_data, total_timesteps, 'pruned',
                                 val_env=val_env, train_env=train_env,
                                 selection_metric=metric_name)
            raise

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            self._register_trial(trial, hyperparams, fold_data, total_timesteps, 'failed',
                                 metrics={'error': str(e)[:200]},
                                 val_env=val_env, train_env=train_env,
                                 selection_metric=metric_name)
            raise

        finally:
            for env in (train_env, val_env):
                if env is not None:
                    env.close()
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def optimize(
        self,
        stock_symbols: Optional[list] = None,
        fold: Optional[str] = None,
        total_timesteps: int = 100_000,
        *,
        split_spec: Optional[str] = None,
        price_norm: str = 'stats',
        price_norm_macd: bool = False,
        env_settings: Optional[Dict[str, Any]] = None,
        show_progress_bar: bool = True,
        panel_df: Optional[pd.DataFrame] = None,
    ) -> optuna.Study:
        """
        Hiper parametre optimizasyonu calistirir (Faz 9 / R1).

        Args:
            stock_symbols: None = RL panelinin TUM sembolleri (rotanin evreni). Alt
                kume verilirse arama o evrende yapilir; rota baska evrenle egitirse
                kaydi `hpo_universe_mismatch` ile acik kalir.
            fold: `evaluation/splits.py` kat adi. None = tanimin ILK kati: o katin
                dogrulamasi butun katlarin testinden once biter, bulunan ayar her
                katta temiz kullanilabilir (tasarim varsayimi; kat basina HPO da
                kurali saglar, maliyeti kat sayisi kadar).
            total_timesteps: deneme basina egitim adimi.
            split_spec: bolme tanimi adi (varsayilan 'wf-v1').
            price_norm / price_norm_macd: rotadaki kolla AYNI olmali (K25).
            env_settings: initial_balance, commission_rate, max_shares_per_trade
                (varsayilanlar egitim isteginin varsayilanlari).
            panel_df: gostergeli ham panel (test icin; verilmezse dosyadan).

        Tarih pencereleri artik istekten ALINMAZ: eski imza (train_start, ...)
        kaldirildi. Test blogu burada hic okunmaz.
        """
        from evaluation import rl_folds
        from evaluation.reproducibility import resolve_seed
        from evaluation.splits import get_spec

        spec = get_spec(split_spec)
        fold_def = spec.fold(fold) if fold is not None else spec.folds[0]
        self.price_norm, self.price_norm_macd = price_norm, price_norm_macd
        # Ayar tohumu kayda girer; egitim tohumu bundan FARKLI olmali (Eimer vd. 2023) —
        # rota `hpo_lineage` ile olcer. Calismanin butun denemeleri ayni tohumu paylasir.
        self.seed = resolve_seed(self.seed)

        rl_panel = rl_folds.load_rl_panel(price_norm=price_norm, df=panel_df)
        self.indicator_spec = rl_panel.indicator_spec
        fold_data = rl_folds.build_fold(rl_panel, spec, fold_def, symbols=stock_symbols)
        fundamental_df, macro_df = self._phase2_frames(fold_data.symbols)
        self.env_settings = {'initial_balance': 1_000_000, 'commission_rate': 0.001,
                             'max_shares_per_trade': 100, **(env_settings or {}),
                             'phase': self.phase, 'reward_type': self.reward_type,
                             'fundamental_df': fundamental_df, 'macro_df': macro_df}
        val_dates = sorted(pd.unique(pd.to_datetime(fold_data.val.index.get_level_values('date'))))
        risk_free = rl_folds.load_risk_free(val_dates)
        self.fold_report = fold_data.report

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Hyperparameter Optimization for {self.algorithm_name.upper()}")
        logger.info(f"Kat {spec.name}/{fold_data.fold}: {fold_data.report['ranges']}")
        logger.info(f"Semboller: {len(fold_data.symbols)} | price_norm={price_norm} | "
                    f"tohum {self.seed} | Trials: {self.n_trials}, Jobs: {self.n_jobs}")
        logger.info(f"{'='*80}\n")

        # Create study
        self.study = self.create_study()

        # Track total optimization time
        optimization_start = time.time()

        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, fold_data, total_timesteps, risk_free),
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
            # Faz 9 / R1: ayarin hangi kat / bolme / tohumla arandigi
            'fold_report': getattr(self, 'fold_report', None),
            'seed': self.seed,
            'price_norm': getattr(self, 'price_norm', None),
            'all_trials': trial_details,
        }

        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"✅ Best parameters saved to {filepath}")
        logger.info(f"   Saved {len(trial_details)} trial details with timing information")
