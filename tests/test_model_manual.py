"""
Manuel model test - saved model ile test set üzerinde evaluation
"""
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import make_env
from data.data_fetcher import DataFetcher

print("=" * 80)
print("MANUAL MODEL TEST")
print("=" * 80)

# Load data
fetcher = DataFetcher()
df = fetcher.load_data('stock_data_with_indicators.csv')
train_df, val_df, test_df = fetcher.split_data(df)

print(f"\nTest set: {len(test_df)} rows")

# Create test environment
def make_test_env():
    return make_env(
        df=test_df,
        initial_balance=1000000,
        commission_rate=0.001,
        max_shares_per_trade=100
    )

test_env = DummyVecEnv([make_test_env])

# Load model
model_path = "models/sac_phase1_20251112_233832.zip"
print(f"\nLoading model: {model_path}")
model = SAC.load(model_path)

# Run test
print("\nRunning test evaluation...")
print(f"Environment instance before reset: {id(test_env.envs[0])}")
obs = test_env.reset()
print(f"Environment instance after reset: {id(test_env.envs[0])}")

done = np.array([False])
step_count = 0

# Track the actual environment instance
actual_env = test_env.envs[0]

metrics = None
while not done[0]:
    action, _states = model.predict(obs, deterministic=False)

    # CRITICAL: Save state BEFORE step (which may autoreset)
    pre_step_trades = len(actual_env.trades_history)
    pre_step_trades_copy = actual_env.trades_history.copy()
    pre_step_portfolio_values = actual_env.portfolio_values.copy()

    obs, reward, done, info = test_env.step(action)
    step_count += 1

    # IMMEDIATELY restore if autoreset happened (trades_history cleared)
    if len(actual_env.trades_history) == 0 and pre_step_trades > 0:
        print(f"\n⚠️  Autoreset detected! Restoring {pre_step_trades} trades")
        actual_env.trades_history = pre_step_trades_copy
        actual_env.portfolio_values = pre_step_portfolio_values

    if step_count % 50 == 0:
        print(f"  Step {step_count}...")

    if done[0]:
        print(f"\n✅ Episode done at step {step_count}")
        break

print(f"\nTest completed: {step_count} steps")
print(f"Environment instance after test: {id(test_env.envs[0])}")

# Get metrics from restored state
print(f"\nGetting metrics from restored state...")
print(f"Getting metrics from: {id(actual_env)}")
metrics = actual_env.get_metrics()

# DEBUG: Check if trades_history is populated
print(f"DEBUG: trades_history length = {len(actual_env.trades_history)}")
print(f"DEBUG: portfolio_values length = {len(actual_env.portfolio_values)}")

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)
print(f"Total trades:          {metrics['total_trades']}")
print(f"Cumulative return:     {metrics['cumulative_return']:.2%}")
print(f"Sharpe ratio:          {metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] else "Sharpe ratio:          N/A")
print(f"Max drawdown:          {metrics['max_drawdown']:.2%}")
print(f"Final portfolio value: ${metrics['final_portfolio_value']:,.2f}")
print("=" * 80)

if metrics['total_trades'] == 0:
    print("\n❌ PROBLEM: Still 0 trades in test set!")
    print("\nDebugging info:")
    print(f"  - Test env created: OK")
    print(f"  - Model loaded: OK")
    print(f"  - Steps executed: {step_count}")
    print(f"  - Environment should have 299 days")
else:
    print(f"\n✅ SUCCESS: {metrics['total_trades']} trades executed!")
