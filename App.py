"""
app.py — Sales Forecasting Streamlit App
Run locally : streamlit run app.py
Deploy      : push to GitHub → connect Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from model import (
    preprocess,
    add_time_features,
    run_prophet,
    evaluate,
    get_forecast_summary,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3a3f6e;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label {
        color: #8b92b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a5b4fc;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #13151f; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: #c7d0f8 !important;
    }

    /* Divider */
    hr { border-color: #2d3155; }

    /* DataFrame */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.45);
    }

    /* Info boxes */
    .info-box {
        background: #1a1f35;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #c7d0f8;
        font-size: 0.9rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Sales Forecasting")
    st.markdown("---")

    st.markdown("### 📁 Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Use Sample Dataset", "Upload CSV"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"],
            help="Must have at least a Date column and a numeric Sales column.",
        )

    st.markdown("---")
    st.markdown("### ⚙️ Column Mapping")
    date_col_input  = st.text_input("Date column name",  value="Date")
    sales_col_input = st.text_input("Sales column name", value="Sales")

    st.markdown("---")
    st.markdown("### 🔮 Forecast Settings")

    forecast_horizon = st.slider(
        "Forecast horizon (days)", min_value=30, max_value=365, value=90, step=30
    )
    freq = st.selectbox(
        "Data frequency",
        options=["MS", "W", "D"],
        format_func=lambda x: {"MS": "Monthly", "W": "Weekly", "D": "Daily"}[x],
        index=0,
    )
    seasonality_mode = st.selectbox(
        "Seasonality mode",
        ["multiplicative", "additive"],
        index=0,
    )
    changepoint_scale = st.slider(
        "Trend flexibility", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        help="Higher = more flexible trend (risk of overfitting)."
    )
    yearly  = st.checkbox("Yearly seasonality",  value=True)
    weekly  = st.checkbox("Weekly seasonality",  value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:10px 0 4px 0'>
  <h1 style='color:#e2e8f0;font-size:2rem;font-weight:800;margin:0'>
    📊 Sales Forecasting Dashboard
  </h1>
  <p style='color:#8b92b8;font-size:1rem;margin-top:4px'>
    Time-series analysis & future sales prediction using Facebook Prophet
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ── Helper: load data ─────────────────────────────────────────────────────────
@st.cache_data
def load_sample():
    return pd.read_csv("data.csv")


def get_dataframe():
    if data_source == "Upload CSV" and uploaded_file:
        return pd.read_csv(uploaded_file)
    return load_sample()


# ── Helper: Plotly theme ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,17,23,0.6)",
    font=dict(color="#c7d0f8", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1e2540", linecolor="#2d3155"),
    yaxis=dict(gridcolor="#1e2540", linecolor="#2d3155"),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
)


# ── Main flow ─────────────────────────────────────────────────────────────────
raw_df = get_dataframe()

# Validate columns
if date_col_input not in raw_df.columns or sales_col_input not in raw_df.columns:
    st.error(
        f"Columns **'{date_col_input}'** or **'{sales_col_input}'** not found. "
        f"Available columns: {list(raw_df.columns)}"
    )
    st.stop()

# Preprocess
df = preprocess(raw_df, date_col_input, sales_col_input)
df_feat = add_time_features(df, date_col_input)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Data Overview", "📈 Trend Analysis", "🔮 Forecast", "📊 Evaluation"]
)


# ══════════════════════════════════════════════
# TAB 1 — Data Overview
# ══════════════════════════════════════════════
with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("📅 Total Records",   f"{len(df):,}")
    col_b.metric("📆 Date Range",      f"{df[date_col_input].dt.year.min()} – {df[date_col_input].dt.year.max()}")
    col_c.metric("💰 Total Sales",     f"${df[sales_col_input].sum():,.0f}")
    col_d.metric("📈 Avg Monthly Sale",f"${df[sales_col_input].mean():,.0f}")

    st.markdown("")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-header">🗃️ Raw Dataset</p>', unsafe_allow_html=True)
        st.dataframe(
            df_feat.style.format({sales_col_input: "${:,.0f}"}),
            use_container_width=True,
            height=340,
        )

    with col_right:
        st.markdown('<p class="section-header">📊 Descriptive Statistics</p>', unsafe_allow_html=True)
        stats = df[sales_col_input].describe().rename({
            "count": "Count", "mean": "Mean", "std": "Std Dev",
            "min": "Min", "25%": "25th %ile", "50%": "Median",
            "75%": "75th %ile", "max": "Max",
        })
        stats_df = pd.DataFrame({"Metric": stats.index, "Value": stats.values.round(2)})
        stats_df["Value"] = stats_df["Value"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(stats_df, use_container_width=True, hide_index=True, height=340)

    # Download raw data
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Processed Data",
        data=csv_bytes,
        file_name="processed_sales.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════
# TAB 2 — Trend Analysis
# ══════════════════════════════════════════════
with tab2:
    # Sales over time
    st.markdown('<p class="section-header">📈 Sales Over Time</p>', unsafe_allow_html=True)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df[date_col_input], y=df[sales_col_input],
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=6, color="#818cf8"),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.12)",
    ))
    # 3-period rolling average
    df["rolling_avg"] = df[sales_col_input].rolling(3, min_periods=1).mean()
    fig_trend.add_trace(go.Scatter(
        x=df[date_col_input], y=df["rolling_avg"],
        mode="lines",
        name="3-Period Moving Avg",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    ))
    fig_trend.update_layout(title="Historical Sales Trend", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Monthly seasonality
        st.markdown('<p class="section-header">🗓️ Monthly Seasonality</p>', unsafe_allow_html=True)
        monthly = df_feat.groupby("month")[sales_col_input].mean().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x-1])
        fig_month = go.Figure(go.Bar(
            x=monthly["month_name"], y=monthly[sales_col_input],
            marker=dict(
                color=monthly[sales_col_input],
                colorscale="Viridis",
                showscale=False,
            ),
        ))
        fig_month.update_layout(title="Avg Sales by Month", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        # Yearly comparison
        st.markdown('<p class="section-header">📅 Yearly Comparison</p>', unsafe_allow_html=True)
        yearly_df = df_feat.groupby("year")[sales_col_input].sum().reset_index()
        fig_year = go.Figure(go.Bar(
            x=yearly_df["year"].astype(str),
            y=yearly_df[sales_col_input],
            marker=dict(
                color=yearly_df[sales_col_input],
                colorscale="Blues",
                showscale=False,
            ),
            text=yearly_df[sales_col_input].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
        ))
        fig_year.update_layout(title="Total Sales by Year", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_year, use_container_width=True)

    # Growth rate
    st.markdown('<p class="section-header">📊 Period-over-Period Growth Rate (%)</p>', unsafe_allow_html=True)
    df["growth"] = df[sales_col_input].pct_change() * 100
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Bar(
        x=df[date_col_input],
        y=df["growth"],
        marker_color=df["growth"].apply(lambda x: "#4ade80" if x >= 0 else "#f87171"),
        name="Growth %",
    ))
    fig_growth.add_hline(y=0, line_color="#475569", line_dash="dot")
    fig_growth.update_layout(title="Sales Growth Rate", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_growth, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — Forecast
# ══════════════════════════════════════════════
with tab3:
    if not run_btn:
        st.markdown("""
        <div class='info-box'>
        ⚡ Configure your forecast settings in the <b>sidebar</b> and click
        <b>🚀 Run Forecast</b> to generate predictions.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🤖 Training Prophet model — this may take a few seconds…"):
            try:
                model, forecast = run_prophet(
                    df,
                    date_col=date_col_input,
                    sales_col=sales_col_input,
                    periods=forecast_horizon,
                    freq=freq,
                    seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly,
                    weekly_seasonality=weekly,
                    changepoint_prior_scale=changepoint_scale,
                )
                st.session_state["model"]    = model
                st.session_state["forecast"] = forecast
                st.success("✅ Forecast generated successfully!")
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()

    if "forecast" in st.session_state:
        forecast = st.session_state["forecast"]
        model    = st.session_state["model"]

        # KPI strip
        future_fc = forecast.tail(forecast_horizon)
        col1, col2, col3 = st.columns(3)
        col1.metric("🔮 Forecast Period",     f"{forecast_horizon} days")
        col2.metric("💰 Predicted Total",     f"${future_fc['yhat'].sum():,.0f}")
        col3.metric("📈 Avg Daily Forecast",  f"${future_fc['yhat'].mean():,.0f}")

        st.markdown("")

        # Main forecast chart
        st.markdown('<p class="section-header">🔮 Sales Forecast with Confidence Interval</p>', unsafe_allow_html=True)
        fig_fc = go.Figure()

        # Confidence band
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Confidence Band",
        ))
        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="#818cf8", width=2.5),
        ))
        # Actual sales
        fig_fc.add_trace(go.Scatter(
            x=df[date_col_input], y=df[sales_col_input],
            mode="markers+lines",
            name="Actual Sales",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=7, color="#fbbf24"),
        ))
        # Forecast start marker
        last_actual = df[date_col_input].max()
        fig_fc.add_vline(
            x=last_actual, line_dash="dash",
            line_color="#94a3b8", line_width=1.5,
            annotation_text="Forecast Start",
            annotation_font_color="#94a3b8",
        )
        fig_fc.update_layout(
            title=f"Sales Forecast — Next {forecast_horizon} Days",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast components
        st.markdown('<p class="section-header">🧩 Forecast Components</p>', unsafe_allow_html=True)
        comp_cols = ["ds", "trend"]
        if "yearly" in forecast.columns:
            comp_cols.append("yearly")
        if "weekly" in forecast.columns:
            comp_cols.append("weekly")

        fig_comp = make_subplots(
            rows=len(comp_cols) - 1, cols=1,
            subplot_titles=[c.capitalize() for c in comp_cols[1:]],
            vertical_spacing=0.12,
        )
        colors = ["#6366f1", "#f59e0b", "#34d399", "#f87171"]
        for i, comp in enumerate(comp_cols[1:], 1):
            fig_comp.add_trace(
                go.Scatter(
                    x=forecast["ds"], y=forecast[comp],
                    mode="lines", name=comp.capitalize(),
                    line=dict(color=colors[i - 1], width=2),
                ),
                row=i, col=1,
            )
        fig_comp.update_layout(
            height=250 * (len(comp_cols) - 1),
            showlegend=False,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,17,23,0.6)",
            font=dict(color="#c7d0f8"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Forecast table + download
        st.markdown('<p class="section-header">📋 Forecast Table</p>', unsafe_allow_html=True)
        summary = get_forecast_summary(forecast, forecast_horizon)
        st.dataframe(summary.style.format({
            "Forecast": "${:,.2f}",
            "Lower Bound": "${:,.2f}",
            "Upper Bound": "${:,.2f}",
        }), use_container_width=True, height=280)

        csv_fc = summary.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv_fc,
            file_name="sales_forecast.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════
# TAB 4 — Evaluation
# ══════════════════════════════════════════════
with tab4:
    if "forecast" not in st.session_state:
        st.markdown("""
        <div class='info-box'>
        ℹ️ Run the forecast first (Tab 3) to see model evaluation metrics.
        </div>
        """, unsafe_allow_html=True)
    else:
        forecast = st.session_state["forecast"]

        metrics = evaluate(df, forecast, date_col_input, sales_col_input)

        # Metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "📉 MAE",
            f"${metrics['MAE']:,.2f}" if metrics["MAE"] else "N/A",
            help="Mean Absolute Error — avg absolute difference between actual and predicted.",
        )
        col2.metric(
            "📐 RMSE",
            f"${metrics['RMSE']:,.2f}" if metrics["RMSE"] else "N/A",
            help="Root Mean Squared Error — penalises large errors more than MAE.",
        )
        col3.metric(
            "📊 MAPE",
            f"{metrics['MAPE']:.2f}%" if metrics["MAPE"] else "N/A",
            help="Mean Absolute Percentage Error — scale-independent accuracy metric.",
        )

        st.markdown("")

        # Actual vs Predicted scatter
        st.markdown('<p class="section-header">🔍 Actual vs Predicted</p>', unsafe_allow_html=True)
        merged = df[[date_col_input, sales_col_input]].merge(
            forecast[["ds", "yhat"]].rename(columns={"ds": date_col_input}),
            on=date_col_input,
            how="inner",
        )
        if not merged.empty:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=merged[sales_col_input], y=merged["yhat"],
                mode="markers",
                marker=dict(color="#6366f1", size=10, opacity=0.8,
                            line=dict(color="#fff", width=1)),
                name="Predicted vs Actual",
            ))
            mn = merged[[sales_col_input, "yhat"]].min().min()
            mx = merged[[sales_col_input, "yhat"]].max().max()
            fig_scatter.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                line=dict(color="#f59e0b", dash="dash", width=1.5),
                name="Perfect Fit",
            ))
            fig_scatter.update_layout(
                title="Actual vs Predicted Sales",
                xaxis_title="Actual Sales ($)",
                yaxis_title="Predicted Sales ($)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Residuals
            st.markdown('<p class="section-header">📉 Residuals (Actual − Predicted)</p>', unsafe_allow_html=True)
            merged["residual"] = merged[sales_col_input] - merged["yhat"]
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Bar(
                x=merged[date_col_input],
                y=merged["residual"],
                marker_color=merged["residual"].apply(
                    lambda x: "#4ade80" if x >= 0 else "#f87171"
                ),
                name="Residual",
            ))
            fig_resid.add_hline(y=0, line_color="#94a3b8", line_dash="dot")
            fig_resid.update_layout(title="Residual Plot", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_resid, use_container_width=True)
        else:
            st.info("Not enough overlapping data to plot actuals vs predicted.")

        # Interpretation guide
        st.markdown("---")
        st.markdown('<p class="section-header">📖 Metric Guide</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        <b>MAE (Mean Absolute Error)</b> — Average absolute error in the same units as sales.
        Lower = better.<br><br>
        <b>RMSE (Root Mean Squared Error)</b> — Similar to MAE but penalises large errors more
        heavily. Always ≥ MAE.<br><br>
        <b>MAPE (Mean Absolute Percentage Error)</b> — Scale-free metric as a %.
        &lt;10% = excellent, 10–20% = good, &gt;20% = needs improvement.
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#475569;font-size:0.8rem;padding:10px 0'>
  📈 Sales Forecasting Dashboard · Built with Streamlit & Prophet ·
  <a href='https://github.com' style='color:#6366f1'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)