"""
app.py — Sales Forecasting App
Uses ONLY: streamlit, pandas, numpy, altair (all pre-installed on Streamlit Cloud)
No prophet, no statsmodels, no scikit-learn needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1e2130,#252840);
    border:1px solid #3a3f6e; border-radius:12px;
    padding:16px 20px; box-shadow:0 4px 15px rgba(0,0,0,.3);
}
[data-testid="metric-container"] label { color:#8b92b8!important; font-size:.85rem!important; font-weight:600!important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e2e8f0!important; font-size:1.6rem!important; font-weight:700!important; }
.section-header { font-size:1rem; font-weight:700; color:#a5b4fc; letter-spacing:.04em; margin-bottom:.4rem; text-transform:uppercase; }
[data-testid="stSidebar"] { background-color:#13151f; }
.info-box { background:#1a1f35; border-left:4px solid #6366f1; border-radius:8px; padding:14px 18px; margin:10px 0; color:#c7d0f8; font-size:.9rem; line-height:1.6; }
.stButton>button { background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white; border:none; border-radius:8px; padding:.55rem 1.4rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Forecasting Engine (pure numpy) ──────────────────────────────────────────

def linear_forecast(y: np.ndarray, steps: int):
    """Weighted linear regression trend extrapolation."""
    n = len(y)
    x = np.arange(n, dtype=float)
    w = np.linspace(0.3, 1.0, n)          # recent points get more weight
    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)
    b1 = np.sum(w * (x - xw) * (y - yw)) / np.sum(w * (x - xw)**2)
    b0 = yw - b1 * xw
    x_future = np.arange(n, n + steps, dtype=float)
    return b0 + b1 * x_future


def seasonal_factors(y: np.ndarray, period: int = 12):
    """Compute multiplicative seasonal indices."""
    if len(y) < period * 2:
        return np.ones(period)
    n_full = (len(y) // period) * period
    y_trim = y[:n_full].reshape(-1, period)
    row_means = y_trim.mean(axis=1, keepdims=True)
    row_means[row_means == 0] = 1
    factors = (y_trim / row_means).mean(axis=0)
    factors /= factors.mean()          # normalise so they average to 1
    return factors


def forecast_series(y: np.ndarray, steps: int, freq: str):
    """Trend × seasonal decomposition forecast with 80% confidence band."""
    period = {"MS": 12, "W": 52, "D": 7}.get(freq, 12)

    trend_fc = linear_forecast(y, steps)

    # seasonal factor for each future step
    factors = seasonal_factors(y, min(period, len(y) // 2 or 1))
    start_idx = len(y) % len(factors)
    sf = np.array([factors[(start_idx + i) % len(factors)] for i in range(steps)])

    yhat = trend_fc * sf
    yhat = np.maximum(yhat, 0)

    # residual std for confidence interval
    trend_in = linear_forecast(y[:-steps] if steps < len(y) else y, len(y))
    resid_std = np.std(y - trend_in[:len(y)]) if len(y) > 3 else np.std(y) * 0.1
    margin = 1.28 * resid_std             # ~80% CI

    return yhat, yhat - margin, yhat + margin


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_CSV = """Date,Sales
2022-01-01,45200
2022-02-01,41800
2022-03-01,49300
2022-04-01,52100
2022-05-01,54700
2022-06-01,58200
2022-07-01,61400
2022-08-01,59800
2022-09-01,63200
2022-10-01,71500
2022-11-01,89400
2022-12-01,102300
2023-01-01,48600
2023-02-01,44100
2023-03-01,53800
2023-04-01,57200
2023-05-01,60100
2023-06-01,63900
2023-07-01,67300
2023-08-01,65100
2023-09-01,69800
2023-10-01,78200
2023-11-01,96700
2023-12-01,114500
2024-01-01,52400
2024-02-01,47900
2024-03-01,58100
2024-04-01,62700
2024-05-01,66800
2024-06-01,70400
2024-07-01,73900
2024-08-01,71200
2024-09-01,76500
2024-10-01,85300
2024-11-01,105200
2024-12-01,124800"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Sales Forecasting")
    st.markdown("---")
    st.markdown("### 📁 Data Source")
    src = st.radio("Source", ["Use Sample Dataset", "Upload CSV"], label_visibility="collapsed")
    uploaded = None
    if src == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ Column Mapping")
    date_col  = st.text_input("Date column",  "Date")
    sales_col = st.text_input("Sales column", "Sales")

    st.markdown("---")
    st.markdown("### 🔮 Forecast Settings")
    horizon = st.slider("Forecast horizon (days)", 30, 365, 90, 30)
    freq    = st.selectbox("Data frequency",
                           ["MS","W","D"],
                           format_func=lambda x: {"MS":"Monthly","W":"Weekly","D":"Daily"}[x])
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding:10px 0 4px 0'>
  <h1 style='color:#e2e8f0;font-size:2rem;font-weight:800;margin:0'>
    📊 Sales Forecasting Dashboard
  </h1>
  <p style='color:#8b92b8;font-size:1rem;margin-top:4px'>
    Time-series analysis & future sales prediction
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_sample():
    return pd.read_csv(io.StringIO(SAMPLE_CSV))

if src == "Upload CSV" and uploaded:
    raw_df = pd.read_csv(uploaded)
else:
    raw_df = load_sample()

if date_col not in raw_df.columns or sales_col not in raw_df.columns:
    st.error(f"Columns '{date_col}' or '{sales_col}' not found. Available: {list(raw_df.columns)}")
    st.stop()

df = raw_df[[date_col, sales_col]].copy()
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).dropna().drop_duplicates(subset=[date_col]).reset_index(drop=True)
df["year"]    = df[date_col].dt.year
df["month"]   = df[date_col].dt.month
df["quarter"] = df[date_col].dt.quarter
df["rolling"] = df[sales_col].rolling(3, min_periods=1).mean()
df["growth"]  = df[sales_col].pct_change() * 100


# ── Altair theme helper ───────────────────────────────────────────────────────

def chart_props(title=""):
    return {
        "title": alt.TitleParams(title, color="#c7d0f8"),
        "background": "transparent",
        "config": alt.Config(
            axis=alt.AxisConfig(gridColor="#1e2540", labelColor="#8b92b8", titleColor="#8b92b8"),
            view=alt.ViewConfig(strokeOpacity=0),
            legend=alt.LegendConfig(labelColor="#c7d0f8", titleColor="#c7d0f8"),
        ),
    }


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview","📈 Trend Analysis","🔮 Forecast","📊 Evaluation"])


# ════════════════════ TAB 1 — Data Overview ════════════════════

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📅 Records",     f"{len(df):,}")
    c2.metric("📆 Date Range",  f"{df[date_col].dt.year.min()} – {df[date_col].dt.year.max()}")
    c3.metric("💰 Total Sales", f"${df[sales_col].sum():,.0f}")
    c4.metric("📈 Avg Sale",    f"${df[sales_col].mean():,.0f}")

    st.markdown("")
    cl, cr = st.columns([3,2])

    with cl:
        st.markdown('<p class="section-header">🗃️ Dataset</p>', unsafe_allow_html=True)
        st.dataframe(df[[date_col,sales_col,"year","month","quarter"]],
                     use_container_width=True, height=340)

    with cr:
        st.markdown('<p class="section-header">📊 Statistics</p>', unsafe_allow_html=True)
        s = df[sales_col].describe()
        stats = pd.DataFrame({
            "Metric": ["Count","Mean","Std Dev","Min","25th %ile","Median","75th %ile","Max"],
            "Value":  [f"${v:,.2f}" for v in s.values]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True, height=340)

    st.download_button("⬇️ Download Data", df.to_csv(index=False).encode(),
                       "sales_data.csv", "text/csv")


# ════════════════════ TAB 2 — Trend Analysis ════════════════════

with tab2:
    # Trend chart
    st.markdown('<p class="section-header">📈 Sales Trend</p>', unsafe_allow_html=True)
    base = alt.Chart(df).encode(x=alt.X(f"{date_col}:T", title="Date"))
    area = base.mark_area(opacity=0.15, color="#6366f1").encode(y=alt.Y(f"{sales_col}:Q", title="Sales ($)"))
    line = base.mark_line(color="#818cf8", strokeWidth=2.5).encode(y=f"{sales_col}:Q")
    avg  = base.mark_line(color="#f59e0b", strokeWidth=2, strokeDash=[6,3]).encode(y="rolling:Q")
    st.altair_chart((area + line + avg).properties(height=280, **chart_props("Historical Sales + 3-Period Moving Avg")),
                    use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">🗓️ Monthly Seasonality</p>', unsafe_allow_html=True)
        m_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = df.groupby("month")[sales_col].mean().reset_index()
        monthly["month_name"] = monthly["month"].map(lambda x: m_names[x-1])
        chart = alt.Chart(monthly).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("month_name:N", sort=m_names, title="Month"),
            y=alt.Y(f"{sales_col}:Q", title="Avg Sales ($)"),
            color=alt.Color(f"{sales_col}:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["month_name:N", alt.Tooltip(f"{sales_col}:Q", format="$,.0f")]
        ).properties(height=260, **chart_props("Avg Sales by Month"))
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">📅 Yearly Comparison</p>', unsafe_allow_html=True)
        yearly = df.groupby("year")[sales_col].sum().reset_index()
        chart2 = alt.Chart(yearly).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(f"{sales_col}:Q", title="Total Sales ($)"),
            color=alt.Color(f"{sales_col}:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["year:O", alt.Tooltip(f"{sales_col}:Q", format="$,.0f")]
        ).properties(height=260, **chart_props("Total Sales by Year"))
        st.altair_chart(chart2, use_container_width=True)

    # Growth rate
    st.markdown('<p class="section-header">📊 Period-over-Period Growth %</p>', unsafe_allow_html=True)
    df_g = df.dropna(subset=["growth"]).copy()
    df_g["color"] = df_g["growth"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    growth_chart = alt.Chart(df_g).mark_bar().encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y("growth:Q", title="Growth %"),
        color=alt.Color("color:N", scale=alt.Scale(domain=["Positive","Negative"],
                        range=["#4ade80","#f87171"]), legend=None),
        tooltip=[f"{date_col}:T", alt.Tooltip("growth:Q", format=".1f")]
    ).properties(height=220, **chart_props("Sales Growth Rate (%)"))
    st.altair_chart(growth_chart, use_container_width=True)


# ════════════════════ TAB 3 — Forecast ════════════════════

with tab3:
    if not run_btn:
        st.markdown("<div class='info-box'>⚡ Set your forecast settings in the sidebar and click <b>🚀 Run Forecast</b>.</div>",
                    unsafe_allow_html=True)
    else:
        with st.spinner("🤖 Running forecast…"):
            y = df[sales_col].values.astype(float)
            freq_steps = {"MS": max(1, horizon//30), "W": max(1, horizon//7), "D": horizon}
            steps = freq_steps.get(freq, max(1, horizon//30))

            yhat, yhat_lo, yhat_hi = forecast_series(y, steps, freq)

            # Build future dates
            last_date = df[date_col].max()
            if freq == "MS":
                future_dates = pd.date_range(last_date, periods=steps+1, freq="MS")[1:]
            elif freq == "W":
                future_dates = pd.date_range(last_date, periods=steps+1, freq="W")[1:]
            else:
                future_dates = pd.date_range(last_date, periods=steps+1, freq="D")[1:]

            fc_df = pd.DataFrame({
                "Date": future_dates,
                "Forecast": yhat,
                "Lower": np.maximum(yhat_lo, 0),
                "Upper": yhat_hi,
            })
            st.session_state["fc_df"] = fc_df
            st.success("✅ Forecast ready!")

    if "fc_df" in st.session_state:
        fc_df = st.session_state["fc_df"]

        c1,c2,c3 = st.columns(3)
        c1.metric("🔮 Periods Forecast",  f"{len(fc_df)}")
        c2.metric("💰 Predicted Total",   f"${fc_df['Forecast'].sum():,.0f}")
        c3.metric("📈 Avg Period Forecast",f"${fc_df['Forecast'].mean():,.0f}")
        st.markdown("")

        # Build combined chart
        hist_chart_df = df[[date_col, sales_col]].rename(columns={date_col:"Date", sales_col:"Value"})
        hist_chart_df["Type"] = "Actual"

        fc_line_df = fc_df[["Date","Forecast"]].rename(columns={"Forecast":"Value"})
        fc_line_df["Type"] = "Forecast"

        combined = pd.concat([hist_chart_df, fc_line_df], ignore_index=True)

        band = alt.Chart(fc_df).mark_area(opacity=0.2, color="#6366f1").encode(
            x="Date:T", y="Lower:Q", y2="Upper:Q"
        )
        lines = alt.Chart(combined).mark_line(strokeWidth=2.5).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Sales ($)"),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["Actual","Forecast"], range=["#f59e0b","#818cf8"])),
            tooltip=["Date:T","Type:N", alt.Tooltip("Value:Q", format="$,.0f")]
        )
        vline = alt.Chart(pd.DataFrame({"x":[df[date_col].max()]})).mark_rule(
            color="#94a3b8", strokeDash=[4,4]
        ).encode(x="x:T")

        st.markdown('<p class="section-header">🔮 Sales Forecast with Confidence Interval</p>',
                    unsafe_allow_html=True)
        st.altair_chart((band + lines + vline).properties(height=320,
                        **chart_props(f"Sales Forecast — Next {horizon} Days")),
                        use_container_width=True)

        # Table
        st.markdown('<p class="section-header">📋 Forecast Table</p>', unsafe_allow_html=True)
        display_fc = fc_df.copy()
        display_fc["Date"] = display_fc["Date"].dt.strftime("%Y-%m-%d")
        display_fc[["Forecast","Lower","Upper"]] = display_fc[["Forecast","Lower","Upper"]].round(2)
        st.dataframe(display_fc, use_container_width=True, hide_index=True, height=260)
        st.download_button("⬇️ Download Forecast CSV",
                           display_fc.to_csv(index=False).encode(),
                           "forecast.csv","text/csv")


# ════════════════════ TAB 4 — Evaluation ════════════════════

with tab4:
    if "fc_df" not in st.session_state:
        st.markdown("<div class='info-box'>ℹ️ Run the forecast first (Tab 3) to see evaluation.</div>",
                    unsafe_allow_html=True)
    else:
        # In-sample fit
        y = df[sales_col].values.astype(float)
        n = len(y)
        x = np.arange(n, dtype=float)
        w = np.linspace(0.3,1.0,n)
        xw = np.average(x, weights=w); yw = np.average(y, weights=w)
        b1 = np.sum(w*(x-xw)*(y-yw)) / np.sum(w*(x-xw)**2)
        b0 = yw - b1*xw
        y_pred = b0 + b1*x

        mae  = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        mape = np.mean(np.abs((y - y_pred) / np.where(y==0,1,y)))*100

        c1,c2,c3 = st.columns(3)
        c1.metric("📉 MAE",  f"${mae:,.2f}")
        c2.metric("📐 RMSE", f"${rmse:,.2f}")
        c3.metric("📊 MAPE", f"{mape:.2f}%")
        st.markdown("")

        # Actual vs Predicted scatter
        eval_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})
        scatter = alt.Chart(eval_df).mark_circle(size=80, color="#6366f1", opacity=0.8).encode(
            x=alt.X("Actual:Q", title="Actual Sales ($)"),
            y=alt.Y("Predicted:Q", title="Predicted Sales ($)"),
            tooltip=[alt.Tooltip("Actual:Q",format="$,.0f"), alt.Tooltip("Predicted:Q",format="$,.0f")]
        )
        mn = min(y.min(), y_pred.min()); mx = max(y.max(), y_pred.max())
        line_df = pd.DataFrame({"x":[mn,mx],"y":[mn,mx]})
        perf_line = alt.Chart(line_df).mark_line(color="#f59e0b", strokeDash=[6,3]).encode(
            x="x:Q", y="y:Q")

        st.markdown('<p class="section-header">🔍 Actual vs Predicted</p>', unsafe_allow_html=True)
        st.altair_chart((scatter + perf_line).properties(height=280,
                        **chart_props("Actual vs Predicted Sales")),
                        use_container_width=True)

        # Residuals
        st.markdown('<p class="section-header">📉 Residuals</p>', unsafe_allow_html=True)
        res_df = df[[date_col]].copy()
        res_df["residual"] = y - y_pred
        res_df["color"] = res_df["residual"].apply(lambda x: "Pos" if x>=0 else "Neg")
        resid_chart = alt.Chart(res_df).mark_bar().encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("residual:Q", title="Residual ($)"),
            color=alt.Color("color:N", scale=alt.Scale(domain=["Pos","Neg"],
                            range=["#4ade80","#f87171"]), legend=None),
            tooltip=[f"{date_col}:T", alt.Tooltip("residual:Q", format="$,.0f")]
        ).properties(height=220, **chart_props("Residual Plot"))
        st.altair_chart(resid_chart, use_container_width=True)

        st.markdown("""
        <div class='info-box'>
        <b>MAE</b> — Avg absolute error in sales units. Lower = better.<br><br>
        <b>RMSE</b> — Penalises large errors more than MAE.<br><br>
        <b>MAPE</b> — % error. &lt;10% excellent · 10–20% good · &gt;20% review model.
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#475569;font-size:.8rem'>📈 Sales Forecasting Dashboard · Built with Streamlit</div>",
            unsafe_allow_html=True)