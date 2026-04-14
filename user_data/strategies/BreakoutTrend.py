# --- BreakoutTrend Strategy v1 ---
# Spot-only trend following: enter on confirmed strength, sit in cash otherwise.
#
# Entry: Bollinger Band squeeze breakout (volatility compression → expansion)
# Filters: BTC 4h trend + pair trend + volume surge + ADX trend strength
# Exit: ATR trailing stop (adapts to each pair's volatility)
# Philosophy: few trades, let winners run, cut losers fast via ATR stop

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, informative
from pandas import DataFrame
from datetime import datetime, timedelta
import talib.abstract as ta
import numpy as np


class BreakoutTrend(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    startup_candle_count = 200
    process_only_new_candles = True

    # Fixed stoploss as safety net (ATR-based custom stoploss is primary)
    stoploss = -0.10
    use_custom_stoploss = True

    # Disable ROI — let the trailing ATR stop manage exits
    minimal_roi = {"0": 100}

    trailing_stop = False

    @property
    def protections(self):
        return [
            {"method": "StoplossGuard", "lookback_period": 24, "trade_limit": 3, "stop_duration": 12, "only_per_pair": False},
            {"method": "MaxDrawdown", "lookback_period": 48, "max_allowed_drawdown": 0.10, "trade_limit": 5, "stop_duration": 24},
            {"method": "LowProfitPairs", "lookback_period": 72, "trade_limit": 2, "stop_duration": 24, "required_profit": -0.02, "only_per_pair": True},
            {"method": "CooldownPeriod", "stop_duration": 4},
        ]

    # Hyperopt parameters
    buy_adx_min = IntParameter(15, 35, default=20, space="buy", optimize=True)
    buy_bb_squeeze = DecimalParameter(0.02, 0.10, default=0.06, space="buy", decimals=2, optimize=True)
    buy_volume_mult = DecimalParameter(1.0, 2.5, default=1.2, space="buy", decimals=1, optimize=True)
    atr_sl_mult = DecimalParameter(2.0, 5.0, default=3.5, space="buy", decimals=1, optimize=True)

    # ── BTC market regime (4h) ─────────────────────────────────────────────
    @informative("4h", "BTC/USDT")
    def populate_indicators_btc_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        ema_cross = dataframe["ema21"] > dataframe["ema50"]
        ema50_rising = dataframe["ema50"] > dataframe["ema50"].shift(5)
        dataframe["btc_bull"] = (ema_cross & ema50_rising).astype(int)
        return dataframe

    # ── 1h indicators ──────────────────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands (squeeze detection + breakout)
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_middle"] = bb["middleband"]
        dataframe["bb_width"] = (bb["upperband"] - bb["lowerband"]) / bb["middleband"]

        # Track minimum BB width over last 20 candles (squeeze detection)
        dataframe["bb_width_min20"] = dataframe["bb_width"].rolling(20).min()

        # Trend indicators
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ATR for adaptive stop-loss
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Volume relative to 20-period SMA
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_sma20"]

        # Donchian channel (N-period high breakout)
        dataframe["high_20"] = dataframe["high"].rolling(20).max()

        return dataframe

    # ── EMA-based trailing stoploss ───────────────────────────────────────
    # In a trend, price stays above EMA21. Trail stop below it with ATR buffer.
    def custom_stoploss(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last = dataframe.iloc[-1]
        ema21 = last.get("ema21", 0)
        atr = last.get("atr", 0)

        if ema21 <= 0 or atr <= 0 or current_rate <= 0:
            return self.stoploss

        # Stop level = EMA21 - (1 ATR buffer below EMA21)
        stop_price = ema21 - (atr * 1.0)
        stop_distance = (current_rate - stop_price) / current_rate

        # Only tighten from default stoploss, never widen
        if stop_distance > abs(self.stoploss):
            return self.stoploss

        return -stop_distance

    # ── Custom exit: regime change + stale trade cleanup ───────────────────
    def custom_exit(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs,
    ) -> str | bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False

        last = dataframe.iloc[-1]
        elapsed = current_time - trade.open_date_utc

        # Regime change: BTC turned bearish, exit losing trades immediately
        if last["btc_usdt_btc_bull_4h"] != 1 and current_profit < 0:
            return "regime_change"

        # Pair trend broke down (price below EMA50): exit if losing
        if last["close"] < last["ema50"] and current_profit < 0 and elapsed > timedelta(hours=4):
            return "trend_break"

        # Stale trade: no momentum after 48h
        if abs(current_profit) < 0.005 and elapsed > timedelta(hours=48):
            return "timeout_flat"

        return False

    # ── Entry: breakout on confirmed strength ──────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Gate 1: BTC market regime must be bullish
        btc_bull = dataframe["btc_usdt_btc_bull_4h"] == 1

        # Gate 2: pair-specific trend confirmation
        pair_uptrend = dataframe["ema21"] > dataframe["ema50"]
        pair_above_ema = dataframe["close"] > dataframe["ema50"]
        adx_strong = dataframe["adx"] > self.buy_adx_min.value

        # Gate 3: volume sanity
        volume_surge = dataframe["volume_ratio"] > self.buy_volume_mult.value
        has_volume = dataframe["volume"] > 0

        # Signal A: BB squeeze breakout
        # BB width near its 20-period minimum = compressed volatility
        # Price breaks above upper band = expansion begins
        bb_compressed = dataframe["bb_width"] <= dataframe["bb_width_min20"] * (1 + self.buy_bb_squeeze.value)
        bb_breakout = dataframe["close"] > dataframe["bb_upper"]

        sig_squeeze = (
            btc_bull & pair_uptrend & pair_above_ema & adx_strong &
            bb_compressed & bb_breakout &
            volume_surge & has_volume
        )

        # Signal B: Donchian breakout (new 20-period high with volume)
        # ADX not required here — making a new high IS the confirmation
        new_high = dataframe["close"] >= dataframe["high_20"]

        sig_donchian = (
            btc_bull & pair_uptrend & pair_above_ema &
            new_high &
            volume_surge & has_volume
        )

        dataframe.loc[sig_squeeze | sig_donchian, "enter_long"] = 1
        dataframe.loc[sig_squeeze, "enter_tag"] = "bb_squeeze_breakout"
        dataframe.loc[sig_donchian, "enter_tag"] = "donchian_breakout"

        return dataframe

    # ── Exit signal: trend exhaustion ──────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit when pair trend reverses (EMA21 crosses below EMA50)
        dataframe.loc[
            (dataframe["ema21"] < dataframe["ema50"]),
            "exit_long"
        ] = 1
        return dataframe
