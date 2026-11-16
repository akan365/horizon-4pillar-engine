from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    session_tz: str
    sess_start: tuple[int, int]
    sess_end: tuple[int, int]
    ib_minutes: int = 30
    point_value: float = 1.0
    spread_divisor: float = 100.0
    slippage_pts: float = 0.50


@dataclass
class PillarToggles:
    use_pillar1_regime: bool = True      # Macro/Trend Regime (EMA + ATR bounds; uses 'cot_bias' if column exists)
    use_pillar2_session: bool = True     # High-Liquidity Session & Initial Balance Window
    use_pillar3_structure: bool = True   # Value Retest (IB Edge / VWAP / 'hvn_level' if column exists)
    use_pillar4_orderflow: bool = True   # Absorption Confirmation (CLV + RelVol / real tick 'cvd_delta' if present)


@dataclass
class RiskConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.015
    max_drawdown_halt: float = 0.10
    target_r: float = 2.0
    min_ib_atr: float = 0.45
    max_ib_atr: float = 2.00
    max_stop_atr: float = 1.25
    ema_span: int = 200
    vol_mult: float = 1.15
    clv_thresh: float = 0.25


@dataclass
class TradeRecord:
    symbol: str
    session_date: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    units: float
    risk_dollars: float
    pnl: float
    r_multiple: float
    exit_reason: str


def detect_and_resample_ticks(
    raw_df: pd.DataFrame, bar_freq: str = "5min", spread_divisor: float = 100.0
) -> tuple[pd.DataFrame, bool]:
    """
    Detects whether `raw_df` is raw tick data (bid/ask or price/side) or standard OHLCV bars.
    If tick data, aggregates it causally into OHLCV bars while preserving true tick microstructure:
      - Exact average spread in points
      - True Volume Delta (`cvd_delta`) from aggressor side or Lee-Ready tick rule
      - Tick count as volume fallback if volume is missing
    Returns (standardized_ohlcv_df, was_tick_data).
    """
    df = raw_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    time_col = next(
        (c for c in ("time", "timestamp", "datetime", "date", "utc", "gmt time") if c in df.columns),
        df.columns[0],
    )
    df["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")

    has_ohlc = all(k in df.columns for k in ("open", "high", "low", "close"))
    has_bid_ask = "bid" in df.columns and "ask" in df.columns
    has_tick_price = any(k in df.columns for k in ("price", "last", "close", "bid"))

    if has_ohlc and not has_bid_ask:
        if "volume" not in df.columns:
            df["volume"] = 1.0
        return df.reset_index(), False

    if has_bid_ask:
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        mid = 0.5 * (bid + ask)
        tick_spread = (ask - bid) * spread_divisor
    elif has_tick_price:
        p_col = next(k for k in ("price", "last", "close", "bid") if k in df.columns)
        mid = pd.to_numeric(df[p_col], errors="coerce")
        tick_spread = (
            pd.to_numeric(df["spread"], errors="coerce")
            if "spread" in df.columns
            else pd.Series(0.0, index=df.index)
        )
    else:
        raise ValueError(
            "Unsupported file format. Expected OHLCV columns or tick columns (time, bid, ask / price, volume)."
        )

    valid = mid.notna()
    mid = mid[valid]
    tick_spread = tick_spread[valid].fillna(0.0)
    df = df[valid]

    if "bid_volume" in df.columns and "ask_volume" in df.columns:
        t_vol = pd.to_numeric(df["bid_volume"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["ask_volume"], errors="coerce"
        ).fillna(0.0)
    elif "volume" in df.columns:
        t_vol = pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
    elif "vol" in df.columns:
        t_vol = pd.to_numeric(df["vol"], errors="coerce").fillna(1.0)
    else:
        t_vol = pd.Series(1.0, index=df.index)

    if "side" in df.columns:
        s = df["side"].astype(str).str.lower()
        sign = np.where(
            s.str.startswith(("b", "1", "+")), 1.0, np.where(s.str.startswith(("s", "-")), -1.0, 0.0)
        )
    else:
        diff = mid.diff().fillna(0.0)
        sign = np.sign(diff).replace(0.0, np.nan).ffill().fillna(0.0)

    signed_vol = t_vol * sign

    ohlc = mid.resample(bar_freq).ohlc()
    vol_bar = t_vol.resample(bar_freq).sum()
    spread_bar = tick_spread.resample(bar_freq).mean()
    delta_bar = signed_vol.resample(bar_freq).sum()

    bars = ohlc.dropna(subset=["open", "close"]).copy()
    bars["volume"] = vol_bar.reindex(bars.index).fillna(0.0)
    bars["spread"] = spread_bar.reindex(bars.index).fillna(0.0)
    bars["cvd_delta"] = delta_bar.reindex(bars.index).fillna(0.0)
    return bars.reset_index(), True


class HorizonFourPillarEngine:
    """
    Modular Four-Pillar AND Gate Auction Engine supporting both OHLCV bars and
    raw Tick CSVs (auto-aggregated to intraday bars with real bid-ask spread & tick CVD).
    """

    def __init__(self, spec: InstrumentSpec, risk: RiskConfig, pillars: PillarToggles):
        self.spec = spec
        self.risk = risk
        self.pillars = pillars

    def prepare_features(self, df: pd.DataFrame, atr_span: int = 20) -> pd.DataFrame:
        data = df.copy()
        data.columns = [str(c).lower() for c in data.columns]
        data["time"] = pd.to_datetime(data["time"], utc=True)
        data = data.set_index("time").sort_index()

        data["ema"] = data["close"].ewm(
            span=self.risk.ema_span, adjust=False, min_periods=self.risk.ema_span
        ).mean()
        prev_close = data["close"].shift(1)
        tr = np.maximum(
            data["high"] - data["low"],
            np.maximum(np.abs(data["high"] - prev_close), np.abs(data["low"] - prev_close)),
        )
        data["atr"] = tr.rolling(atr_span, min_periods=atr_span).mean()

        rng = (data["high"] - data["low"]).replace(0.0, np.nan)
        data["clv"] = (((data["close"] - data["low"]) - (data["high"] - data["close"])) / rng).fillna(0.0)
        data["rel_vol"] = data["volume"] / data["volume"].rolling(20, min_periods=20).mean()
        return data

    def run(self, raw_df: pd.DataFrame) -> dict:
        df = self.prepare_features(raw_df)
        local = df.index.tz_convert(self.spec.session_tz)
        mod = local.hour * 60 + local.minute

        start_m = self.spec.sess_start[0] * 60 + self.spec.sess_start[1]
        end_m = self.spec.sess_end[0] * 60 + self.spec.sess_end[1]
        sess_len = (end_m - start_m) % 1440 or 1440

        off = (mod - start_m) % 1440
        in_sess = (off < sess_len) if self.pillars.use_pillar2_session else np.ones(len(df), dtype=bool)
        anchor = local - pd.to_timedelta(off, unit="min")
        sess_dates = anchor.tz_localize(None).normalize().date

        equity = self.risk.initial_capital
        peak_equity = equity
        trades: list[TradeRecord] = []

        for d, grp in df[in_sess].groupby(sess_dates[in_sess]):
            if (peak_equity - equity) / peak_equity >= self.risk.max_drawdown_halt:
                break

            grp_local = grp.index.tz_convert(self.spec.session_tz)
            grp_off = ((grp_local.hour * 60 + grp_local.minute) - start_m) % 1440

            ib_bars = grp[grp_off < self.spec.ib_minutes] if self.pillars.use_pillar2_session else grp.iloc[:2]
            post_bars = (
                grp[(grp_off >= self.spec.ib_minutes) & (grp_off < sess_len - 15)]
                if self.pillars.use_pillar2_session
                else grp.iloc[2:-1]
            )
            if len(ib_bars) == 0 or len(post_bars) < 4:
                continue

            ib_high = float(ib_bars["high"].max())
            ib_low = float(ib_bars["low"].min())
            ib_mid = 0.5 * (ib_high + ib_low)
            ib_rng = ib_high - ib_low
            atr_val = float(ib_bars["atr"].iloc[-1])

            if np.isnan(atr_val) or atr_val <= 0.0 or ib_rng <= 0.0:
                continue

            if self.pillars.use_pillar1_regime:
                if not (self.risk.min_ib_atr * atr_val <= ib_rng <= self.risk.max_ib_atr * atr_val):
                    continue

            cum_pv = float((ib_bars["close"] * ib_bars["volume"]).sum())
            cum_v = float(ib_bars["volume"].sum())

            price_step = max(atr_val * 0.15, 1e-5)
            vol_profile: dict[float, float] = {}
            for b_px, b_vol in zip(ib_bars["close"], ib_bars["volume"]):
                bucket = round(float(b_px) / price_step) * price_step
                vol_profile[bucket] = vol_profile.get(bucket, 0.0) + float(b_vol)

            state = 0
            bias = 0
            pos_side = 0
            entry_px = stop_px = target_px = units = risk_dollars = 0.0
            entry_ts = None
            daily_pnl = 0.0
            traded_today = False

            bars = list(post_bars.itertuples())
            for i, bar in enumerate(bars):
                cum_pv += bar.close * bar.volume
                cum_v += bar.volume
                vwap = cum_pv / cum_v if cum_v > 0 else ib_mid

                bucket = round(float(bar.close) / price_step) * price_step
                vol_profile[bucket] = vol_profile.get(bucket, 0.0) + float(bar.volume)
                session_hvn = max(vol_profile.items(), key=lambda kv: kv[1])[0] if vol_profile else ib_mid

                spread_pts = (bar.spread / self.spec.spread_divisor) if hasattr(bar, "spread") else 0.0
                half_spread = 0.5 * spread_pts
                slip = self.spec.slippage_pts

                if state == 2:
                    exit_px = None
                    reason = ""
                    if pos_side == 1:
                        if bar.low <= stop_px:
                            exit_px, reason = stop_px - slip, "STOP"
                        elif bar.high >= target_px:
                            exit_px, reason = target_px - slip, "TARGET"
                        elif i == len(bars) - 1:
                            exit_px, reason = bar.close - half_spread - slip, "EOD"
                    else:
                        if bar.high >= stop_px:
                            exit_px, reason = stop_px + slip, "STOP"
                        elif bar.low <= target_px:
                            exit_px, reason = target_px + slip, "TARGET"
                        elif i == len(bars) - 1:
                            exit_px, reason = bar.close + half_spread + slip, "EOD"

                    if exit_px is not None:
                        pnl = pos_side * (exit_px - entry_px) * units * self.spec.point_value
                        equity += pnl
                        daily_pnl += pnl
                        peak_equity = max(peak_equity, equity)
                        trades.append(
                            TradeRecord(
                                symbol=self.spec.symbol,
                                session_date=str(d),
                                side="LONG" if pos_side == 1 else "SHORT",
                                entry_time=entry_ts,
                                exit_time=bar.Index,
                                entry_price=round(entry_px, 5),
                                exit_price=round(exit_px, 5),
                                stop_price=round(stop_px, 5),
                                target_price=round(target_px, 5),
                                units=round(units, 4),
                                risk_dollars=round(risk_dollars, 2),
                                pnl=round(pnl, 2),
                                r_multiple=round(pnl / risk_dollars, 3),
                                exit_reason=reason,
                            )
                        )
                        state = 0
                        traded_today = True
                    continue

                if traded_today or i >= len(bars) - 2:
                    continue
                if daily_pnl <= -self.risk.initial_capital * self.risk.max_daily_loss:
                    continue

                cot_ok_long = getattr(bar, "cot_bias", 1) >= 0
                cot_ok_short = getattr(bar, "cot_bias", -1) <= 0
                p1_long = (bar.close > bar.ema and cot_ok_long) if self.pillars.use_pillar1_regime else True
                p1_short = (bar.close < bar.ema and cot_ok_short) if self.pillars.use_pillar1_regime else True
                p4_vol = (bar.rel_vol >= self.risk.vol_mult) if self.pillars.use_pillar4_orderflow else True

                if state == 0:
                    if bar.close > ib_high and p1_long and p4_vol:
                        if not self.pillars.use_pillar3_structure:
                            nxt = bars[i + 1]
                            entry_px = nxt.open + half_spread + slip
                            stop_px = ib_mid - half_spread
                            stop_dist = entry_px - stop_px
                            if 0 < stop_dist <= self.risk.max_stop_atr * atr_val:
                                risk_dollars = equity * self.risk.risk_per_trade
                                units = risk_dollars / (stop_dist * self.spec.point_value)
                                target_px = entry_px + self.risk.target_r * stop_dist
                                pos_side, state, entry_ts = 1, 2, nxt.Index
                        else:
                            state, bias = 1, 1

                    elif bar.close < ib_low and p1_short and p4_vol:
                        if not self.pillars.use_pillar3_structure:
                            nxt = bars[i + 1]
                            entry_px = nxt.open - half_spread - slip
                            stop_px = ib_mid + half_spread
                            stop_dist = stop_px - entry_px
                            if 0 < stop_dist <= self.risk.max_stop_atr * atr_val:
                                risk_dollars = equity * self.risk.risk_per_trade
                                units = risk_dollars / (stop_dist * self.spec.point_value)
                                target_px = entry_px - self.risk.target_r * stop_dist
                                pos_side, state, entry_ts = -1, 2, nxt.Index
                        else:
                            state, bias = 1, -1

                elif state == 1:
                    if (bias == 1 and bar.close < ib_mid) or (bias == -1 and bar.close > ib_mid):
                        state, bias = 0, 0
                        continue

                    if bias == 1:
                        ext_hvn = getattr(bar, "hvn_level", np.nan)
                        struct_ref = ext_hvn if not np.isnan(ext_hvn) else session_hvn
                        value_lvl = max(ib_high, vwap, struct_ref)
                        touched_struct = bar.low <= value_lvl and bar.close > ib_high
                        cvd_ok = getattr(bar, "cvd_delta", 1.0) > 0
                        absorbed = (
                            (bar.clv >= self.risk.clv_thresh and cvd_ok)
                            if self.pillars.use_pillar4_orderflow
                            else True
                        )

                        if touched_struct and absorbed:
                            nxt = bars[i + 1]
                            entry_px = nxt.open + half_spread + slip
                            stop_px = min(bar.low, ib_mid) - half_spread
                            stop_dist = entry_px - stop_px
                            if 0 < stop_dist <= self.risk.max_stop_atr * atr_val:
                                risk_dollars = equity * self.risk.risk_per_trade
                                units = risk_dollars / (stop_dist * self.spec.point_value)
                                target_px = entry_px + self.risk.target_r * stop_dist
                                pos_side, state, entry_ts = 1, 2, nxt.Index

                    elif bias == -1:
                        ext_hvn = getattr(bar, "hvn_level", np.nan)
                        struct_ref = ext_hvn if not np.isnan(ext_hvn) else session_hvn
                        value_lvl = min(ib_low, vwap, struct_ref)
                        touched_struct = bar.high >= value_lvl and bar.close < ib_low
                        cvd_ok = getattr(bar, "cvd_delta", -1.0) < 0
                        absorbed = (
                            (bar.clv <= -self.risk.clv_thresh and cvd_ok)
                            if self.pillars.use_pillar4_orderflow
                            else True
                        )

                        if touched_struct and absorbed:
                            nxt = bars[i + 1]
                            entry_px = nxt.open - half_spread - slip
                            stop_px = max(bar.high, ib_mid) + half_spread
                            stop_dist = stop_px - entry_px
                            if 0 < stop_dist <= self.risk.max_stop_atr * atr_val:
                                risk_dollars = equity * self.risk.risk_per_trade
                                units = risk_dollars / (stop_dist * self.spec.point_value)
                                target_px = entry_px - self.risk.target_r * stop_dist
                                pos_side, state, entry_ts = -1, 2, nxt.Index

        tdf = pd.DataFrame([t.__dict__ for t in trades])
        return {"equity": round(equity, 2), "trades": tdf}