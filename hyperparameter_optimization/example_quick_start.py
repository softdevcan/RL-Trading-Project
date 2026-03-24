"""
Quick Start Example - Hyperparameter Optimization

Bu script, hiper parametre optimizasyonunun nasıl kullanılacağını gösterir.

Kullanım:
    python hyperparameter_optimization/example_quick_start.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from hyperparameter_optimization import (
    PPOOptimizer,
    A2COptimizer,
    get_search_space,
    print_search_space_info,
)
from data.bist30_symbols import PHASE1_SYMBOLS


def example_1_view_search_space():
    """
    Örnek 1: Arama uzayını görüntüleme
    """
    print("\n" + "="*80)
    print("ÖRNEK 1: Arama Uzayını Görüntüleme")
    print("="*80)

    # PPO'nun arama uzayını görüntüle
    print_search_space_info("ppo")


def example_2_quick_optimization():
    """
    Örnek 2: Hızlı test optimizasyonu (5 dakika)
    """
    print("\n" + "="*80)
    print("ÖRNEK 2: Hızlı Test Optimizasyonu")
    print("="*80)

    # Optimizer oluştur
    optimizer = PPOOptimizer(
        n_trials=2,           # Sadece 2 trial (test için)
        n_jobs=1,             # Sequential çalıştırma
        seed=42               # Reproducibility
    )

    # Optimize et
    study = optimizer.optimize(
        stock_symbols=PHASE1_SYMBOLS,             # PHASE1: AKBNK, THYAO, TUPRS, BIMAS, ASELS
        train_start='2023-01-01',
        train_end='2023-06-30',                   # 6 ay training
        val_start='2023-07-01',
        val_end='2023-12-31',                     # 6 ay validation
        total_timesteps=10_000,                    # Kısa training
        eval_freq=2_000,
        n_eval_episodes=3,
        show_progress_bar=True
    )

    # En iyi parametreleri kaydet
    optimizer.save_best_params()

    print("\n✅ Optimizasyon tamamlandı!")
    print(f"En iyi Sharpe Ratio: {study.best_value:.4f}")
    print(f"En iyi parametreler:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


def example_3_production_optimization():
    """
    Örnek 3: Production-grade optimizasyon (6-12 saat)

    NOT: Bu uzun sürer! Sadece gerçek çalışma için kullanın.
    """
    print("\n" + "="*80)
    print("ÖRNEK 3: Production-Grade Optimizasyon")
    print("="*80)
    print("\n⚠️  UYARI: Bu 6-12 saat sürebilir!")
    print("Devam etmek istiyor musunuz? (y/n)")

    response = input().lower()
    if response != 'y':
        print("İptal edildi.")
        return

    # Optimizer oluştur
    optimizer = PPOOptimizer(
        n_trials=50,          # 50 trial
        n_jobs=1,             # GPU memory için sequential
        seed=42
    )

    # Optimize et
    study = optimizer.optimize(
        stock_symbols=PHASE1_SYMBOLS,  # PHASE1: AKBNK, THYAO, TUPRS, BIMAS, ASELS
        train_start='2018-01-01',
        train_end='2022-12-31',   # 3 yıl training
        val_start='2023-01-01',
        val_end='2023-12-31',     # 1 yıl validation
        total_timesteps=100_000,  # Tam training
        eval_freq=5_000,
        n_eval_episodes=5,
        show_progress_bar=True
    )

    # En iyi parametreleri kaydet
    optimizer.save_best_params()

    print("\n✅ Production optimizasyonu tamamlandı!")
    print(f"En iyi Sharpe Ratio: {study.best_value:.4f}")


def example_4_compare_algorithms():
    """
    Örnek 4: Birden fazla algoritmayı karşılaştırma
    """
    print("\n" + "="*80)
    print("ÖRNEK 4: Algoritma Karşılaştırması")
    print("="*80)

    algorithms = {
        'PPO': PPOOptimizer,
        'A2C': A2COptimizer,
    }

    results = {}

    for name, OptimizerClass in algorithms.items():
        print(f"\n{'='*80}")
        print(f"Optimizing: {name}")
        print(f"{'='*80}\n")

        optimizer = OptimizerClass(
            n_trials=2,       # Test için küçük
            n_jobs=1,
            seed=42
        )

        study = optimizer.optimize(
            stock_symbols=PHASE1_SYMBOLS,  # PHASE1: AKBNK, THYAO, TUPRS, BIMAS, ASELS
            train_start='2023-01-01',
            train_end='2023-06-30',
            val_start='2023-07-01',
            val_end='2023-12-31',
            total_timesteps=10_000,
            show_progress_bar=True
        )

        optimizer.save_best_params()
        results[name] = study.best_value

    # Karşılaştırma
    print("\n" + "="*80)
    print("📊 SONUÇLAR:")
    print("="*80)

    for name, best_value in results.items():
        print(f"  {name:10s}: {best_value:.4f}")

    best_algo = max(results, key=results.get)
    print(f"\n🏆 En iyi algoritma: {best_algo} (Sharpe Ratio: {results[best_algo]:.4f})")


def main():
    """
    Ana menü
    """
    print("\n" + "="*100)
    print(" "*30 + "🔬 HYPERPARAMETER OPTIMIZATION - QUICK START")
    print("="*100)
    print("\nÖrnekler:")
    print("  1. Arama uzayını görüntüle")
    print("  2. Hızlı test optimizasyonu (5 dakika)")
    print("  3. Production-grade optimizasyon (6-12 saat) ⚠️")
    print("  4. Algoritma karşılaştırması (10 dakika)")
    print("  0. Çıkış")
    print("="*100)

    choice = input("\nSeçiminiz (0-4): ")

    if choice == "1":
        example_1_view_search_space()
    elif choice == "2":
        example_2_quick_optimization()
    elif choice == "3":
        example_3_production_optimization()
    elif choice == "4":
        example_4_compare_algorithms()
    elif choice == "0":
        print("\nGüle güle! 👋")
    else:
        print("\n❌ Geçersiz seçim!")

    print("\n" + "="*100)
    print("💡 TİP: Tam otomatik çalıştırma için:")
    print("   python hyperparameter_optimization/run_optimization.py --algorithm ppo --trials 50")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
