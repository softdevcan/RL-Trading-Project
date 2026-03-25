"""
Phase 3: Altın Dahil RL Model Eğitimi
BIST-30 Phase 1 hisseleri + GOLD_GRAM_TRY varlığı ile SB3 model eğitir.

Kullanım:
    python scripts/train_with_gold.py --algorithm PPO --timesteps 50000
"""

import sys
import os
import argparse
import logging

# Proje kökünden çalıştırılmalı
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3 Altın RL Eğitimi')
    parser.add_argument('--algorithm', default='PPO',
                        choices=['A2C', 'PPO', 'TD3', 'SAC'],
                        help='RL algoritması')
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='Toplam eğitim timestep sayısı')
    parser.add_argument('--start-date', default='2020-01-01',
                        help='Eğitim verisi başlangıç tarihi')
    parser.add_argument('--initial-balance', type=float, default=1_000_000,
                        help='Başlangıç portföy değeri')
    parser.add_argument('--max-shares', type=int, default=100,
                        help='İşlem başına maksimum hisse')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 3: ALTIN DAHİL RL EGİTİMİ")
    logger.info("=" * 60)
    logger.info(f"Algoritma  : {args.algorithm}")
    logger.info(f"Timesteps  : {args.timesteps:,}")
    logger.info(f"Baslangic  : {args.start_date}")
    logger.info(f"Balance    : {args.initial_balance:,.0f} TRY")

    # --- 1. Veri ---
    from data.bist30_symbols import get_symbols
    from data.data_fetcher import DataFetcher
    from data.gold_fetcher import GoldFetcher
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    import pandas as pd

    stock_symbols = get_symbols(phase=1)
    logger.info(f"\nHisseler    : {stock_symbols}")
    logger.info("Altin varlik: GOLD_GRAM_TRY")

    # Hisse verisi
    logger.info("\n[1/4] Hisse verisi cekiliyor...")
    stock_fetcher = DataFetcher(start_date=args.start_date)
    stock_df = stock_fetcher.fetch_stock_data(stock_symbols, save=False)
    stock_df = stock_fetcher.clean_data(stock_df)

    # Altın verisi
    logger.info("[2/4] Altin verisi cekiliyor...")
    gold_fetcher = GoldFetcher(start_date=args.start_date)
    gold_df = gold_fetcher.fetch_all_gold_data()

    # GOLD_GRAM_TRY'yi çıkart
    gram_gold_df = gold_df.xs('GOLD_GRAM_TRY', level='symbol').copy()
    gram_gold_df.index = pd.to_datetime(gram_gold_df.index).tz_localize(None)
    gram_gold_multi = pd.concat([gram_gold_df], keys=['GOLD_GRAM_TRY'])
    gram_gold_multi.index.names = ['symbol', 'date']

    # Hisse + altın birleştir
    combined_df = pd.concat([stock_df, gram_gold_multi])
    combined_df.index.names = ['symbol', 'date']
    logger.info(f"  Toplam sembol: {combined_df.index.get_level_values('symbol').unique().tolist()}")

    # Teknik indikatörler
    logger.info("[3/4] Teknik indikatörler hesaplanıyor...")
    combined_df = add_indicators_to_multi_symbol_df(combined_df)

    # Train split
    train_df, _, _ = stock_fetcher.split_data(combined_df)

    # --- 2. Environment ---
    logger.info("[4/4] Trading environment olusturuluyor (Phase 3)...")
    from env.trading_env import TradingEnv

    env = TradingEnv(
        df=train_df,
        initial_balance=args.initial_balance,
        max_shares_per_trade=args.max_shares,
        phase=3,
        reward_type='simple',
    )

    logger.info(f"  Semboller      : {env.symbols}")
    logger.info(f"  State boyutu   : {env.observation_space.shape[0]}")
    logger.info(f"  Action boyutu  : {env.action_space.shape[0]}")

    # --- 3. Model ---
    logger.info(f"\nModel olusturuluyor ({args.algorithm})...")

    if args.algorithm == 'A2C':
        from stable_baselines3 import A2C
        model = A2C('MlpPolicy', env, verbose=1)
    elif args.algorithm == 'PPO':
        from stable_baselines3 import PPO
        model = PPO('MlpPolicy', env, verbose=1)
    elif args.algorithm == 'TD3':
        from stable_baselines3 import TD3
        model = TD3('MlpPolicy', env, verbose=1)
    else:  # SAC
        from stable_baselines3 import SAC
        model = SAC('MlpPolicy', env, verbose=1)

    logger.info(f"Egitim basliyor ({args.timesteps:,} timesteps)...")
    model.learn(total_timesteps=args.timesteps)

    # --- 4. Kaydet ---
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.algorithm.lower()}_phase3_{ts}"
    model_path = os.path.join('models', model_name)
    os.makedirs('models', exist_ok=True)
    model.save(model_path)

    logger.info(f"\nModel kaydedildi: {model_path}.zip")
    logger.info("Phase 3 egitimi tamamlandi!")


if __name__ == '__main__':
    main()
