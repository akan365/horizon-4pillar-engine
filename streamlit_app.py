import numpy as np
import pandas as pd
import streamlit as st
from horizon_engine import (
    HorizonFourPillarEngine,
    InstrumentSpec,
    PillarToggles,
    RiskConfig,
    detect_and_resample_ticks,
)

st.set_page_config(page_title="Horizon Capital — 4-Pillar Engine", layout="wide")
st.title("Horizon Capital — Four-Pillar AND Gate Dashboard")

with st.sidebar:
    st.header("1. Four-Pillar AND Gate Toggles")
    p1 = st.toggle(
        "Pillar 1: Macro & Volatility Regime",
        value=True,
        help="EMA trend + ATR range bounds (auto-uses 'cot_bias' if present)",
    )
    p2 = st.toggle(
        "Pillar 2: Liquidity Session Window",
        value=True,
        help="Restricts entries to active exchange session & Initial Balance",
    )
    p3 = st.toggle(
        "Pillar 3: Structural Value Retest",
        value=True,
        help="Waits for pullback to IB Edge / VWAP / Session HVN instead of chasing breakout",
    )
    p4 = st.toggle(
        "Pillar 4: Order-Flow Absorption",
        value=True,
        help="Requires CLV + relative volume + real tick CVD delta (when tick data is uploaded)",
    )

    st.divider()
    st.header("2. Market, Data & Tick Resampling")
    uploaded = st.file_uploader("Upload OHLCV or Raw Tick CSV", type=["csv"])
    tick_bar_freq = st.selectbox(
        "Tick Aggregation Bar Size",
        ["1min", "5min", "15min"],
        index=1,
        help="Used automatically when raw tick data (bid/ask or price) is uploaded",
    )
    preset = st.selectbox("Quick Preset", ["NASDAQ (NAS100)", "Gold (XAUUSD)", "EUR/USD", "Custom"])
    if preset == "NASDAQ (NAS100)":
        def_sym, def_tz, def_sh, def_sm, def_eh, def_em, def_div, def_slip = (
            "NAS100", "America/New_York", 9, 30, 16, 0, 100.0, 0.75
        )
    elif preset == "Gold (XAUUSD)":
        def_sym, def_tz, def_sh, def_sm, def_eh, def_em, def_div, def_slip = (
            "XAUUSD", "America/New_York", 8, 0, 16, 0, 100.0, 0.15
        )
    elif preset == "EUR/USD":
        def_sym, def_tz, def_sh, def_sm, def_eh, def_em, def_div, def_slip = (
            "EURUSD", "Europe/London", 8, 0, 16, 30, 100000.0, 0.00005
        )
    else:
        def_sym, def_tz, def_sh, def_sm, def_eh, def_em, def_div, def_slip = (
            "ASSET", "UTC", 0, 0, 23, 45, 100.0, 0.10
        )

    symbol = st.text_input("Symbol", value=def_sym)
    session_tz = st.selectbox(
        "Exchange Timezone",
        ["America/New_York", "Europe/London", "UTC"],
        index=["America/New_York", "Europe/London", "UTC"].index(def_tz),
    )
    c1, c2 = st.columns(2)
    start_h = c1.number_input("Open Hr", 0, 23, def_sh)
    start_m = c2.number_input("Open Min", 0, 59, def_sm)
    end_h = c1.number_input("Close Hr", 0, 23, def_eh)
    end_m = c2.number_input("Close Min", 0, 59, def_em)
    ib_mins = st.slider("Initial Balance (min)", 15, 60, 30, step=15)

    st.divider()
    st.header("3. Risk & Validation Gates")
    capital = st.number_input("Initial Capital ($)", 1000.0, 1_000_000.0, 10_000.0, step=1000.0)
    risk_pct = st.slider("Risk per Trade (%)", 0.1, 2.0, 0.5, step=0.1) / 100.0
    target_r = st.slider("Target R-Multiple", 1.0, 4.0, 2.0, step=0.2)
    spread_div = st.number_input("Spread Divisor", 1.0, 100000.0, float(def_div))
    slippage = st.number_input("Slippage / Side (pts)", 0.0, 50.0, float(def_slip), format="%.5f")
    oos_only = st.checkbox("Gate 1: Blind 30% Out-of-Sample Only", value=False)
    stress_2x = st.checkbox("Gate 3: 2x Friction Stress Test", value=False)

if uploaded is None:
    st.info(
        "Upload an intraday OHLCV CSV or raw Tick CSV (`time, bid, ask, volume` or `time, price, volume, side`) in the sidebar."
    )
    st.stop()

raw_df = pd.read_csv(uploaded)
try:
    df, is_tick = detect_and_resample_ticks(raw_df, bar_freq=tick_bar_freq, spread_divisor=spread_div)
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

if oos_only:
    df = df.iloc[int(len(df) * 0.70):].reset_index(drop=True)

pillars = PillarToggles(
    use_pillar1_regime=p1,
    use_pillar2_session=p2,
    use_pillar3_structure=p3,
    use_pillar4_orderflow=p4,
)
spec = InstrumentSpec(
    symbol=symbol,
    session_tz=session_tz,
    sess_start=(int(start_h), int(start_m)),
    sess_end=(int(end_h), int(end_m)),
    ib_minutes=int(ib_mins),
    spread_divisor=spread_div / (2.0 if stress_2x else 1.0),
    slippage_pts=slippage * (2.0 if stress_2x else 1.0),
)
risk = RiskConfig(initial_capital=capital, risk_per_trade=risk_pct, target_r=target_r)

res = HorizonFourPillarEngine(spec, risk, pillars).run(df)
tdf = res["trades"]

active_labels = [
    name for name, on in [("P1:Regime", p1), ("P2:Session", p2), ("P3:Structure", p3), ("P4:OrderFlow", p4)] if on
]
feed_badge = (
    f"TICK FEED -> {tick_bar_freq} Bars (True Bid/Ask Spread + Tick CVD Active)"
    if is_tick
    else "OHLCV BAR FEED"
)
st.caption(
    f"Feed: {feed_badge} | Active AND Gate: {' + '.join(active_labels) if active_labels else 'NONE'} | Bars: {len(df):,}"
)

if tdf.empty:
    st.warning("No trades triggered with this Pillar combination. Try toggling off one gate or adjusting session/ATR bounds.")
    st.stop()

wins = tdf[tdf["pnl"] > 0]
losses = tdf[tdf["pnl"] <= 0]
pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) else np.inf
eq_curve = capital + tdf["pnl"].cumsum()
dd_series = ((eq_curve - eq_curve.cummax()) / eq_curve.cummax()) * 100.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Net Profit", f"{(res['equity']/capital - 1)*100:+.2f}%", f"${res['equity'] - capital:+,.2f}")
k2.metric("Win Rate", f"{len(wins)/len(tdf)*100:.1f}%", f"{len(tdf)} trades")
k3.metric("Expectancy", f"{tdf['r_multiple'].mean():+.3f} R")
k4.metric("Profit Factor", f"{pf:.2f}")
k5.metric("Max Drawdown", f"{dd_series.min():.2f}%")

tab1, tab2, tab3 = st.tabs(["Equity & Drawdown", "Trade Log", "Exit Breakdown"])
with tab1:
    idx_ts = pd.to_datetime(tdf["exit_time"])
    st.subheader("Equity Curve ($)")
    st.line_chart(pd.DataFrame({"Equity ($)": eq_curve.values}, index=idx_ts))
    st.subheader("Drawdown (%)")
    st.area_chart(pd.DataFrame({"Drawdown (%)": dd_series.values}, index=idx_ts))
with tab2:
    st.dataframe(tdf, use_container_width=True)
    st.download_button(
        "Download Trade Log CSV", tdf.to_csv(index=False), f"{symbol}_4pillar_trades.csv", "text/csv"
    )
with tab3:
    st.bar_chart(tdf["exit_reason"].value_counts())
