"""
Daily Trading Service
Helper functions for daily trading decisions
"""

import os
import json
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from data.technical_indicators import add_indicators_to_multi_symbol_df

logger = logging.getLogger(__name__)


# ==================== RISK PARAMETERS ====================

def get_risk_parameters(risk_mode: str) -> dict:
    """
    Get risk parameters based on risk mode

    Args:
        risk_mode: 'conservative', 'moderate', or 'aggressive'

    Returns:
        Dictionary with risk parameters
    """
    params = {
        "conservative": {
            "min_signal_threshold": 0.5,
            "max_position_pct": 0.20,
            "max_daily_trades": 2,
            "description": "Konservatif - Yüksek güven eşiği, düşük pozisyon limiti"
        },
        "moderate": {
            "min_signal_threshold": 0.3,
            "max_position_pct": 0.30,
            "max_daily_trades": 3,
            "description": "Orta - Dengeli risk/getiri"
        },
        "aggressive": {
            "min_signal_threshold": 0.1,
            "max_position_pct": 0.40,
            "max_daily_trades": 5,
            "description": "Agresif - Düşük güven eşiği, yüksek pozisyon limiti"
        }
    }

    return params.get(risk_mode, params["moderate"])


# ==================== MARKET DATA FETCHING ====================

async def fetch_latest_market_data(
    symbols: List[str],
    target_date: str,
    lookback_days: int = 30
) -> pd.DataFrame:
    """
    Fetch latest market data for given symbols

    Args:
        symbols: List of stock symbols (e.g., ['ASELS.IS', 'THYAO.IS'])
        target_date: Target date in YYYY-MM-DD format
        lookback_days: Number of days to look back

    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    logger.info(f"Fetching market data for {len(symbols)} symbols, target: {target_date}")

    try:
        # Calculate date range
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer for indicators

        # Fetch data from yfinance
        data_frames = []

        for symbol in symbols:
            try:
                logger.info(f"Downloading {symbol}...")
                ticker_data = yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True  # Fix FutureWarning
                )

                if ticker_data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                # Reset index to get date as column
                ticker_data = ticker_data.reset_index()

                # Rename columns to lowercase first
                ticker_data.columns = [col.lower() if isinstance(col, str) else col for col in ticker_data.columns]

                # Set date as index temporarily for indicators
                ticker_data = ticker_data.set_index('date')

                # Add technical indicators for this symbol
                logger.info(f"Calculating indicators for {symbol}...")
                from data.technical_indicators import TechnicalIndicatorCalculator
                calculator = TechnicalIndicatorCalculator()
                ticker_data = calculator.add_all_indicators(ticker_data)

                # Reset index again and add symbol column
                ticker_data = ticker_data.reset_index()
                ticker_data['symbol'] = symbol

                logger.info(f"Processed {symbol}, final columns: {ticker_data.columns.tolist()}")
                data_frames.append(ticker_data)

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue

        if not data_frames:
            raise ValueError("No market data could be fetched")

        # Combine all symbols
        combined_df = pd.concat(data_frames, ignore_index=True)

        logger.info(f"Combined DataFrame columns BEFORE flatten: {combined_df.columns.tolist()}")
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        logger.info(f"Columns type: {type(combined_df.columns)}")
        logger.info(f"Is MultiIndex: {isinstance(combined_df.columns, pd.MultiIndex)}")

        # Handle MultiIndex columns or tuple columns
        # Check if any column is a tuple (MultiIndex might appear as regular Index with tuple values)
        has_tuple_columns = any(isinstance(col, tuple) for col in combined_df.columns)

        if isinstance(combined_df.columns, pd.MultiIndex) or has_tuple_columns:
            logger.info("Flattening MultiIndex/tuple columns...")
            # Flatten the columns
            # For tuples like ('Date', ''), just take the first element
            # For tuples like ('Close', 'ASELS.IS'), join with underscore
            new_columns = []
            for col in combined_df.columns:
                if isinstance(col, tuple):
                    # Filter out empty strings and join
                    parts = [str(p) for p in col if p != '']
                    new_columns.append('_'.join(parts) if len(parts) > 1 else parts[0] if parts else 'unknown')
                else:
                    new_columns.append(col)
            combined_df.columns = new_columns
            logger.info(f"Flattened columns: {combined_df.columns.tolist()}")

        # Now lowercase all columns
        combined_df.columns = [col.lower() if isinstance(col, str) else col for col in combined_df.columns]
        logger.info(f"Final columns (lowercase): {combined_df.columns.tolist()}")

        logger.info(f"Sample rows:\n{combined_df.head()}")

        # Verify required columns exist
        if 'symbol' not in combined_df.columns:
            raise ValueError(f"'symbol' column missing! Available columns: {combined_df.columns.tolist()}")
        if 'date' not in combined_df.columns:
            raise ValueError(f"'date' column missing! Available columns: {combined_df.columns.tolist()}")

        # Set multi-index (symbol, date)
        combined_df = combined_df.set_index(['symbol', 'date'])

        # Add technical indicators
        logger.info("Calculating technical indicators...")
        combined_df = add_indicators_to_multi_symbol_df(combined_df)

        logger.info(f"Market data fetched successfully: {len(combined_df)} rows")

        return combined_df

    except Exception as e:
        logger.error(f"Failed to fetch market data: {str(e)}", exc_info=True)
        raise


# ==================== STATE BUILDING ====================

def build_live_state(
    balance: float,
    shares_owned: Dict[str, int],
    market_data: pd.DataFrame,
    target_date: str,
    initial_balance: float = 1_000_000,
    max_shares_per_trade: int = 100
) -> np.ndarray:
    """
    Build state vector for inference

    Args:
        balance: Current cash balance
        shares_owned: Dict of {symbol: shares}
        market_data: Market data with indicators
        target_date: Target date
        initial_balance: Initial balance for normalization
        max_shares_per_trade: Max shares for normalization

    Returns:
        State vector (numpy array)
    """
    logger.info(f"Building state for date: {target_date}")

    symbols = list(shares_owned.keys())
    n_stocks = len(symbols)

    # 1. Balance (normalized with log scale)
    balance_norm = np.log(balance / initial_balance + 1)

    # 2. Shares owned (normalized)
    shares_array = np.array([shares_owned.get(symbol, 0) for symbol in symbols])
    shares_norm = shares_array / max_shares_per_trade

    # 3. Market features
    features = []
    target_dt = pd.to_datetime(target_date)

    for symbol in symbols:
        try:
            # Get data for this symbol on target date
            row = market_data.loc[(symbol, target_dt)]

            # OHLCV
            open_price = row.get('open', 0)
            high_price = row.get('high', 0)
            low_price = row.get('low', 0)
            close_price = row.get('close', 0)
            volume = row.get('volume', 0)

            # Technical indicators
            macd = row.get('macd', 0)
            rsi = row.get('rsi', 50)
            cci = row.get('cci', 0)
            adx = row.get('adx', 0)
            turbulence = row.get('turbulence', 0)

            # Normalize (same as training)
            volume_norm = np.log(volume / 1e6 + 1) / 3 if volume > 0 else 0

            features.extend([
                (open_price - 50) / 50,
                (high_price - 50) / 50,
                (low_price - 50) / 50,
                (close_price - 50) / 50,
                volume_norm,
                np.tanh(macd / 0.1),
                (rsi - 50) / 50,
                np.tanh(cci / 100),
                (adx - 25) / 25,
                np.tanh(turbulence / 2)
            ])

        except KeyError:
            logger.warning(f"No data for {symbol} on {target_date}, using zeros")
            features.extend([0] * 10)

    # Combine all parts
    state = np.concatenate([
        [balance_norm],
        shares_norm,
        features
    ]).astype(np.float32)

    logger.info(f"State built: shape={state.shape}")

    return state


# ==================== PRICE HELPERS ====================

def get_current_prices(market_data: pd.DataFrame, target_date: str) -> Dict[str, float]:
    """
    Get current prices for all symbols

    Args:
        market_data: Market data
        target_date: Target date

    Returns:
        Dict of {symbol: price}
    """
    prices = {}
    target_dt = pd.to_datetime(target_date)

    symbols = market_data.index.get_level_values('symbol').unique()

    for symbol in symbols:
        try:
            row = market_data.loc[(symbol, target_dt)]
            prices[symbol] = float(row['close'])
        except KeyError:
            logger.warning(f"No price for {symbol} on {target_date}")
            prices[symbol] = 0.0

    return prices


# ==================== ACTION INTERPRETATION ====================

def interpret_actions_with_risk(
    action: np.ndarray,
    symbols: List[str],
    current_prices: Dict[str, float],
    balance: float,
    shares_owned: Dict[str, int],
    risk_params: dict,
    max_shares_per_trade: int,
    commission_rate: float = 0.001
) -> List[dict]:
    """
    Interpret model actions with risk filtering

    Args:
        action: Raw action from model (shape: n_stocks)
        symbols: List of symbols
        current_prices: Current prices
        balance: Current balance
        shares_owned: Current shares owned
        risk_params: Risk parameters
        max_shares_per_trade: Maximum shares per trade
        commission_rate: Commission rate

    Returns:
        List of trade decisions
    """
    logger.info(f"Interpreting actions with risk mode: {risk_params.get('description', 'N/A')}")

    # Flatten action if needed
    action = np.array(action).flatten()

    if len(action) != len(symbols):
        raise ValueError(f"Action length mismatch: {len(action)} vs {len(symbols)}")

    decisions = []
    trades_executed = 0
    max_trades = risk_params['max_daily_trades']
    min_threshold = risk_params['min_signal_threshold']
    max_position_pct = risk_params['max_position_pct']

    # Calculate current portfolio value
    portfolio_value = balance + sum(
        shares_owned.get(symbol, 0) * current_prices.get(symbol, 0)
        for symbol in symbols
    )

    for i, symbol in enumerate(symbols):
        raw_signal = float(action[i])
        price = current_prices.get(symbol, 0)
        current_shares = shares_owned.get(symbol, 0)

        # Default: HOLD
        decision = {
            "symbol": symbol,
            "action": "HOLD",
            "raw_signal": raw_signal,
            "shares": 0,
            "price": price,
            "cost": 0.0,
            "revenue": 0.0,
            "commission": 0.0,
            "reason": "",
            "executed": False
        }

        # Check if price is available
        if price == 0:
            decision["reason"] = "No price data available"
            decisions.append(decision)
            continue

        # Check signal threshold
        if abs(raw_signal) < min_threshold:
            decision["reason"] = f"Weak signal ({raw_signal:.2f}), below threshold ({min_threshold})"
            decisions.append(decision)
            continue

        # Check max daily trades
        if trades_executed >= max_trades:
            decision["reason"] = f"Daily trade limit reached ({max_trades})"
            decisions.append(decision)
            continue

        # Scale action to shares
        shares_to_trade = int(round(raw_signal * max_shares_per_trade))

        # BUY SIGNAL
        if shares_to_trade > 0:
            # Check position limit
            position_value = (current_shares + shares_to_trade) * price
            if position_value > portfolio_value * max_position_pct:
                # Adjust shares to fit position limit
                max_position_value = portfolio_value * max_position_pct
                max_shares = int((max_position_value - current_shares * price) / price)
                shares_to_trade = max(0, min(shares_to_trade, max_shares))

                if shares_to_trade == 0:
                    decision["reason"] = f"Position limit reached ({max_position_pct*100:.0f}% of portfolio)"
                    decisions.append(decision)
                    continue

            # Check if we have enough balance
            cost = shares_to_trade * price * (1 + commission_rate)

            if cost > balance:
                # Adjust shares to fit balance
                max_affordable = int(balance / (price * (1 + commission_rate)))
                shares_to_trade = min(shares_to_trade, max_affordable)

                if shares_to_trade == 0:
                    decision["reason"] = f"Insufficient balance (need ₺{cost:,.2f}, have ₺{balance:,.2f})"
                    decisions.append(decision)
                    continue

                cost = shares_to_trade * price * (1 + commission_rate)

            commission = shares_to_trade * price * commission_rate

            decision.update({
                "action": "BUY",
                "shares": shares_to_trade,
                "cost": cost,
                "commission": commission,
                "reason": f"Strong buy signal ({raw_signal:.2f})",
                "executed": True
            })

            trades_executed += 1

        # SELL SIGNAL
        elif shares_to_trade < 0:
            shares_to_sell = min(abs(shares_to_trade), current_shares)

            if shares_to_sell == 0:
                decision["reason"] = "No shares to sell"
                decisions.append(decision)
                continue

            revenue = shares_to_sell * price * (1 - commission_rate)
            commission = shares_to_sell * price * commission_rate

            decision.update({
                "action": "SELL",
                "shares": shares_to_sell,
                "revenue": revenue,
                "commission": commission,
                "reason": f"Sell signal ({raw_signal:.2f})",
                "executed": True
            })

            trades_executed += 1

        decisions.append(decision)

    logger.info(f"Generated {trades_executed} executable trades out of {len(symbols)} signals")

    return decisions


# ==================== PORTFOLIO CALCULATION ====================

def calculate_portfolio_value(
    balance: float,
    shares: Dict[str, int],
    prices: Dict[str, float]
) -> dict:
    """
    Calculate portfolio value

    Args:
        balance: Cash balance
        shares: Shares owned
        prices: Current prices

    Returns:
        Portfolio snapshot dict
    """
    portfolio_value = balance + sum(
        shares.get(symbol, 0) * prices.get(symbol, 0)
        for symbol in shares.keys()
    )

    return {
        "balance": balance,
        "shares": shares.copy(),
        "portfolio_value": portfolio_value
    }


def simulate_portfolio_after_trades(
    balance: float,
    shares: Dict[str, int],
    decisions: List[dict]
) -> dict:
    """
    Simulate portfolio after executing decisions

    Args:
        balance: Current balance
        shares: Current shares
        decisions: Trade decisions

    Returns:
        Portfolio snapshot after trades
    """
    new_balance = balance
    new_shares = shares.copy()

    for decision in decisions:
        if not decision["executed"]:
            continue

        symbol = decision["symbol"]

        if decision["action"] == "BUY":
            new_balance -= decision["cost"]
            new_shares[symbol] = new_shares.get(symbol, 0) + decision["shares"]

        elif decision["action"] == "SELL":
            new_balance += decision["revenue"]
            new_shares[symbol] = new_shares.get(symbol, 0) - decision["shares"]

    # Calculate new portfolio value
    portfolio_value = new_balance + sum(
        new_shares.get(symbol, 0) * decision["price"]
        for decision in decisions
        if decision["symbol"] in new_shares
    )

    return {
        "balance": new_balance,
        "shares": new_shares,
        "portfolio_value": portfolio_value
    }


# ==================== DATA PERSISTENCE ====================

def save_daily_decision(
    date: str,
    decisions: List[dict],
    portfolio_before: dict,
    portfolio_after: dict,
    risk_mode: str,
    max_shares_per_trade: int
):
    """
    Save daily decision to JSON file

    Args:
        date: Decision date
        decisions: List of trade decisions
        portfolio_before: Portfolio before trades
        portfolio_after: Portfolio after trades
        risk_mode: Risk mode used
        max_shares_per_trade: Max shares per trade
    """
    logger.info(f"Saving daily decision for {date}")

    os.makedirs('data/live_trading', exist_ok=True)

    decision_file = 'data/live_trading/trade_decisions.json'

    # Load existing decisions
    if os.path.exists(decision_file):
        with open(decision_file, 'r') as f:
            all_decisions = json.load(f)
    else:
        all_decisions = {}

    # Add new decision
    all_decisions[date] = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_before": portfolio_before,
        "decisions": decisions,
        "portfolio_after": portfolio_after,
        "summary": {
            "total_trades": len([d for d in decisions if d["executed"]]),
            "total_commission": sum(d.get("commission", 0) for d in decisions),
            "daily_return_pct": (
                (portfolio_after["portfolio_value"] - portfolio_before["portfolio_value"])
                / portfolio_before["portfolio_value"] * 100
                if portfolio_before["portfolio_value"] > 0 else 0
            ),
            "risk_mode": risk_mode,
            "max_shares_per_trade": max_shares_per_trade
        }
    }

    # Save back
    with open(decision_file, 'w') as f:
        json.dump(all_decisions, f, indent=2)

    logger.info(f"Decision saved to {decision_file}")


def append_to_portfolio_history(
    date: str,
    portfolio_after: dict,
    daily_return_pct: float
):
    """
    Append to portfolio history CSV

    Args:
        date: Date
        portfolio_after: Portfolio snapshot
        daily_return_pct: Daily return percentage
    """
    logger.info(f"Appending to portfolio history: {date}")

    os.makedirs('data/live_trading', exist_ok=True)

    history_file = 'data/live_trading/portfolio_history.csv'

    # Create new row
    new_row = {
        'date': date,
        'balance': portfolio_after['balance'],
        'portfolio_value': portfolio_after['portfolio_value'],
        'daily_return_pct': daily_return_pct
    }

    # Add shares columns
    for symbol, shares in portfolio_after['shares'].items():
        # Clean symbol name for column
        col_name = symbol.replace('.IS', '_shares')
        new_row[col_name] = shares

    # Append or create
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        # Remove existing entry for this date if exists
        df = df[df['date'] != date]
        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([new_row])

    new_df.to_csv(history_file, index=False)
    logger.info(f"Portfolio history updated: {history_file}")


def load_portfolio_history(days: int = 30) -> dict:
    """
    Load portfolio history

    Args:
        days: Number of days to load

    Returns:
        Dict with dates, values, returns, balances
    """
    history_file = 'data/live_trading/portfolio_history.csv'

    if not os.path.exists(history_file):
        return {
            "dates": [],
            "portfolio_values": [],
            "daily_returns": [],
            "balances": []
        }

    df = pd.read_csv(history_file)
    df = df.tail(days)

    return {
        "dates": df['date'].tolist(),
        "portfolio_values": df['portfolio_value'].tolist(),
        "daily_returns": df['daily_return_pct'].tolist(),
        "balances": df['balance'].tolist()
    }
