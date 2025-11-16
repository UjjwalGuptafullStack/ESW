import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Diwali Air Quality Insights",
    page_icon="🎇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Data loading and preparation
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse timestamps
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df["Timestamp"] = df["created_at"]
    elif "Timestamp" in df.columns:
        # fallback
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    else:
        raise ValueError("Expected a 'created_at' or 'Timestamp' column in CSV")
    
    # Filter out test data - only include data from Oct 18, 2025 onwards
    START_DATE_FILTER = pd.to_datetime('2025-10-18', utc=True)
    df = df[df["Timestamp"] >= START_DATE_FILTER].reset_index(drop=True)

    # Numeric conversions aligned with script.py
    def to_num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    df["PM2.5"] = to_num("field4")
    df["PM10"] = to_num("field5")
    df["Temp_C"] = to_num("field2")
    df["Humidity_pct"] = to_num("field3")

    # Treat -1 as invalid
    for c in ["PM2.5", "PM10", "Temp_C", "Humidity_pct"]:
        if c in df.columns:
            df.loc[df[c] == -1.00, c] = np.nan

    # Sort and time parts
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["hour"] = df["Timestamp"].dt.hour
    df["date"] = df["Timestamp"].dt.date
    return df


def aqi_category_pm25(pm25: float) -> str | float:
    if pd.isna(pm25):
        return np.nan
    if pm25 <= 30:
        return "Good"
    elif pm25 <= 60:
        return "Satisfactory"
    elif pm25 <= 90:
        return "Moderate"
    elif pm25 <= 120:
        return "Poor"
    elif pm25 <= 250:
        return "Very Poor"
    else:
        return "Severe"


# -----------------------------
# Sidebar controls
# -----------------------------
DATA_PATH = "feeds.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Couldn't find {DATA_PATH} in the working directory.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    st.caption("Use these to explore the data and focus on Diwali")

    df = load_data(DATA_PATH)
    min_ts = pd.to_datetime(df["Timestamp"].min())
    max_ts = pd.to_datetime(df["Timestamp"].max())
    
    # Enforce minimum date (post-filtering)
    min_allowed = pd.to_datetime('2025-10-18', utc=True)
    if min_ts < min_allowed:
        min_ts = min_allowed

    # Default Diwali date from script.py
    diwali_default = pd.to_datetime("2025-10-20", utc=True)
    diwali_date = st.date_input(
        "Diwali date", value=diwali_default.date(), min_value=min_ts.date(), max_value=max_ts.date()
    )
    DIWALI_DATE = pd.to_datetime(str(diwali_date), utc=True)

    window_days = st.slider("Show ± days around Diwali", min_value=1, max_value=14, value=7)
    smooth_win = st.slider("Smoothing window (hours)", 1, 24, 3)
    pm25_threshold = st.number_input("PM2.5 threshold (µg/m³)", min_value=10, max_value=200, value=60, step=5)
    
    # Data quality indicator
    st.info(f"📊 Data filtered from Oct 18, 2025 onwards\n({len(df):,} records available)")

    # Global date filter
    st.subheader("Date range filter")
    start_date, end_date = st.date_input(
        "Select date range", value=(min_ts.date(), max_ts.date()), min_value=min_ts.date(), max_value=max_ts.date()
    )

# Apply filters
mask_range = (df["Timestamp"] >= pd.to_datetime(str(start_date), utc=True)) & (
    df["Timestamp"] <= pd.to_datetime(str(end_date), utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
)

work = df.loc[mask_range].copy()

# Period labeling
work["Period"] = np.where(
    (work["Timestamp"] < DIWALI_DATE - pd.Timedelta(days=1)), "Pre-Diwali",
    np.where((work["Timestamp"] <= DIWALI_DATE + pd.Timedelta(days=1)), "During Diwali", "Post-Diwali")
)

# AQI categories
work["AQI_Category"] = work["PM2.5"].apply(aqi_category_pm25)

# Optional smoothing for time series
# Use time-based rolling with a datetime index. Series.rolling does not accept `on`,
# so compute on a temporary DataFrame indexed by Timestamp and merge back.
ws = work.sort_values("Timestamp").set_index("Timestamp").copy()
smooth_cols = []
for col in ["PM2.5", "PM10"]:
    if col in ws.columns:
        sname = f"{col}_smoothed"
        ws[sname] = ws[col].rolling(f"{smooth_win}H").mean()
        smooth_cols.append(sname)

if smooth_cols:
    work = work.merge(ws[smooth_cols].reset_index(), on="Timestamp", how="left")

# Focused window around Diwali
focus_mask = (work["Timestamp"] >= DIWALI_DATE - pd.Timedelta(days=window_days)) & (
    work["Timestamp"] <= DIWALI_DATE + pd.Timedelta(days=window_days)
)
focus = work.loc[focus_mask].copy()

# -----------------------------
# Page header and story
# -----------------------------
st.title("🎇 Diwali Air Quality Insights Dashboard")
st.markdown(
    """
    ### Comprehensive Analysis of Particulate Matter During Diwali 2025
    
    This interactive dashboard analyzes air quality data collected during the Diwali festival period, focusing on PM2.5, PM10, 
    temperature, and humidity measurements. The analysis excludes preliminary test data and covers the period from **October 18, 2025 onwards**.
    
    **🔍 Key Research Questions:**
    - How did Diwali festivities impact air quality?
    - What role did meteorological conditions play?
    - When during the day were pollution peaks most severe?
    - How quickly did air quality recover post-festival?
    
    **📊 Use the sidebar controls** to explore different time windows, adjust smoothing parameters, and set pollution thresholds.
    """
)

# Add methodology note
with st.expander("📋 Methodology & Data Sources"):
    st.markdown("""
    **Data Collection Period:** October 18-30, 2025 (excluding test runs)
    
    **Measurement Parameters:**
    - PM2.5 & PM10: Particulate matter concentrations (µg/m³)
    - Temperature: Ambient air temperature (°C)
    - Humidity: Relative humidity (%)
    - Sampling frequency: Continuous monitoring
    
    **Analysis Methods:**
    - Time-series analysis with configurable smoothing
    - Period-based comparison (Pre/During/Post Diwali)
    - AQI categorization per CPCB standards
    - Correlation analysis between meteorological and pollution parameters
    - Cumulative exposure assessment
    
    **Quality Control:** Negative values (-1) treated as missing data
    """)

# Key metrics
colA, colB, colC, colD = st.columns(4)
colA.metric("Total records", f"{len(work):,}")
colB.metric("Date range", f"{work['Timestamp'].min().date()} → {work['Timestamp'].max().date()}")
colC.metric("Mean PM2.5", f"{work['PM2.5'].mean():.1f} µg/m³")
colD.metric("Mean PM10", f"{work['PM10'].mean():.1f} µg/m³")

st.divider()

# 1) Time-Series Trend Analysis
st.header("1) Time-Series Trends around Diwali")
st.caption("Smoothed series shown where available; orange line marks Diwali.")
fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=("PM2.5 over time", "PM10 over time"))

# PM2.5
if "PM2.5" in focus:
    fig_ts.add_trace(
        go.Scatter(x=focus["Timestamp"], y=focus["PM2.5"], name="PM2.5", mode="lines",
                   line=dict(color="crimson", width=1), opacity=0.4),
        row=1, col=1
    )
    if "PM2.5_smoothed" in focus:
        fig_ts.add_trace(
            go.Scatter(x=focus["Timestamp"], y=focus["PM2.5_smoothed"], name="PM2.5 (smoothed)", mode="lines",
                       line=dict(color="red", width=2)),
            row=1, col=1
        )

# PM10
if "PM10" in focus:
    fig_ts.add_trace(
        go.Scatter(x=focus["Timestamp"], y=focus["PM10"], name="PM10", mode="lines",
                   line=dict(color="royalblue", width=1), opacity=0.4),
        row=2, col=1
    )
    if "PM10_smoothed" in focus:
        fig_ts.add_trace(
            go.Scatter(x=focus["Timestamp"], y=focus["PM10_smoothed"], name="PM10 (smoothed)", mode="lines",
                       line=dict(color="blue", width=2)),
            row=2, col=1
        )

# Diwali line
for r in [1, 2]:
    fig_ts.add_vline(x=DIWALI_DATE, line_width=2, line_dash="dash", line_color="orange", row=r, col=1)

fig_ts.update_layout(height=550, showlegend=True)
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown(
    "During Diwali, spikes in particulate matter are common due to fireworks and festive activity.\n"
    "Use the smoothing window to observe underlying trends beyond short-term bursts."
)

st.divider()

# 2) Temperature-PM Relationship
st.header("2) Temperature vs PM relationship")
st.caption("Explore how ambient temperature correlates with particulate matter.")

def scatter_with_trend(df_in: pd.DataFrame, x: str, y: str, color: str = None, title: str = ""):
    fig = px.scatter(df_in, x=x, y=y, color=color, opacity=0.5, trendline=None)
    # Add simple linear trend line (polyfit) if enough points
    valid = df_in[[x, y]].dropna()
    if len(valid) > 10:
        coeffs = np.polyfit(valid[x], valid[y], 1)
        x_line = np.linspace(valid[x].min(), valid[x].max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Linear fit", line=dict(color="red")))
    fig.update_layout(title=title)
    return fig

valid_temp_pm = work.dropna(subset=["Temp_C", "PM2.5"]) if {"Temp_C", "PM2.5"} <= set(work.columns) else pd.DataFrame()
if not valid_temp_pm.empty:
    c1, c2 = st.columns(2)
    c1.plotly_chart(scatter_with_trend(valid_temp_pm, "Temp_C", "PM2.5", title="Temperature vs PM2.5"), use_container_width=True)
    c2.plotly_chart(scatter_with_trend(valid_temp_pm, "Temp_C", "PM10", title="Temperature vs PM10"), use_container_width=True)
else:
    st.info("Insufficient temperature data to compute relationships.")

st.divider()

# 3) Humidity influence + correlation
st.header("3) Humidity influence and correlation")
valid_hum = work.dropna(subset=["Humidity_pct", "PM2.5"]) if {"Humidity_pct", "PM2.5"} <= set(work.columns) else pd.DataFrame()
cols_corr = [c for c in ["PM2.5", "PM10", "Temp_C", "Humidity_pct"] if c in work.columns]
row = st.columns((2, 2))

if not valid_hum.empty:
    fig_hum = scatter_with_trend(valid_hum, "Humidity_pct", "PM2.5", title="Humidity vs PM2.5")
    row[0].plotly_chart(fig_hum, use_container_width=True)

if len(cols_corr) >= 2:
    corr_df = work[cols_corr].dropna().corr()
    fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu", origin="lower")
    fig_corr.update_layout(title="Correlation matrix")
    row[1].plotly_chart(fig_corr, use_container_width=True)
else:
    row[1].info("Not enough data to compute correlations.")

st.divider()

# 4) Diurnal variation
st.header("4) Diurnal variation in PM levels")
hourly = work.groupby("hour")[[c for c in ["PM2.5", "PM10"] if c in work.columns]].mean().reset_index()
fig_diurnal = go.Figure()
if "PM2.5" in hourly:
    fig_diurnal.add_trace(go.Scatter(x=hourly["hour"], y=hourly["PM2.5"], mode="lines+markers", name="PM2.5", line=dict(color="crimson")))
if "PM10" in hourly:
    fig_diurnal.add_trace(go.Scatter(x=hourly["hour"], y=hourly["PM10"], mode="lines+markers", name="PM10", line=dict(color="royalblue")))
fig_diurnal.update_layout(xaxis=dict(dtick=1), title="Average concentration by hour of day")
st.plotly_chart(fig_diurnal, use_container_width=True)

st.divider()

# 5) AQI distribution
st.header("5) AQI category distribution")
aqi_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
aqi_counts = work["AQI_Category"].value_counts().reindex(aqi_order).fillna(0)
colors = {
    "Good": "green",
    "Satisfactory": "lightgreen",
    "Moderate": "yellow",
    "Poor": "orange",
    "Very Poor": "red",
    "Severe": "darkred",
}

c1, c2 = st.columns(2)
fig_bar = px.bar(x=aqi_counts.index, y=aqi_counts.values, color=aqi_counts.index, color_discrete_map=colors,
                 labels={"x": "AQI Category", "y": "Count"}, title="Distribution of AQI categories")
c1.plotly_chart(fig_bar, use_container_width=True)

fig_pie = px.pie(values=aqi_counts.values, names=aqi_counts.index, color=aqi_counts.index, color_discrete_map=colors,
                 title="AQI category proportion")
c2.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# 6) Temperature inversion indicator (dual-axis)
st.header("6) Temperature inversion indicator")
valid_inv = work.dropna(subset=["Temp_C", "PM2.5"]) if {"Temp_C", "PM2.5"} <= set(work.columns) else pd.DataFrame()
if not valid_inv.empty:
    fig_inv = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig_inv.add_trace(go.Scatter(x=valid_inv["Timestamp"], y=valid_inv["Temp_C"], name="Temperature (°C)", line=dict(color="royalblue")), secondary_y=False)
    fig_inv.add_trace(go.Scatter(x=valid_inv["Timestamp"], y=valid_inv["PM2.5"], name="PM2.5 (µg/m³)", line=dict(color="crimson")), secondary_y=True)
    fig_inv.add_vline(x=DIWALI_DATE, line_width=2, line_dash="dash", line_color="orange")
    fig_inv.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig_inv.update_yaxes(title_text="PM2.5 (µg/m³)", secondary_y=True)
    fig_inv.update_layout(title="Temperature vs PM2.5 over time")
    st.plotly_chart(fig_inv, use_container_width=True)
else:
    st.info("Insufficient data for temperature inversion analysis.")

st.divider()

# 7) Pre/During/Post Diwali comparison (boxplots)
st.header("7) Diwali period comparison")
period_data = work.dropna(subset=["PM2.5", "PM10"]) if {"PM2.5", "PM10"} <= set(work.columns) else work.copy()
order = ["Pre-Diwali", "During Diwali", "Post-Diwali"]
c1, c2 = st.columns(2)
if not period_data.empty and "PM2.5" in period_data:
    fig_box1 = px.box(period_data, x="Period", y="PM2.5", category_orders={"Period": order}, color="Period")
    fig_box1.update_layout(title="PM2.5 distribution by period")
    c1.plotly_chart(fig_box1, use_container_width=True)

if not period_data.empty and "PM10" in period_data:
    fig_box2 = px.box(period_data, x="Period", y="PM10", category_orders={"Period": order}, color="Period")
    fig_box2.update_layout(title="PM10 distribution by period")
    c2.plotly_chart(fig_box2, use_container_width=True)

st.divider()

# 8) Full correlation matrix (again on filtered data)
st.header("8) Comprehensive correlation matrix")
if len(cols_corr) >= 2:
    corr_df2 = work[cols_corr].dropna().corr()
    fig_corr2 = px.imshow(corr_df2, text_auto=True, color_continuous_scale="RdBu", origin="lower")
    st.plotly_chart(fig_corr2, use_container_width=True)

st.divider()

# 9) Cumulative exposure (AOT)
st.header("9) Cumulative exposure above threshold")
if "PM2.5" in work:
    exposure = work.dropna(subset=["PM2.5"]).copy()
    exposure["Excess"] = (exposure["PM2.5"] - pm25_threshold).clip(lower=0)
    exposure["CumulativeExposure"] = exposure["Excess"].cumsum()

    fig_aot = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("PM2.5 exposure above threshold", "Cumulative excess PM2.5"))

    fig_aot.add_trace(go.Scatter(x=exposure["Timestamp"], y=exposure["PM2.5"], name="PM2.5", line=dict(color="darkred")), row=1, col=1)
    fig_aot.add_hline(y=pm25_threshold, line_width=2, line_dash="dash", line_color="orange", row=1, col=1)

    # Shade areas above threshold (approx via filled range)
    over = exposure[exposure["PM2.5"] > pm25_threshold]
    fig_aot.add_trace(go.Scatter(x=over["Timestamp"], y=over["PM2.5"], name="Excess area",
                                 fill="tozeroy", mode="none", fillcolor="rgba(255,0,0,0.2)"), row=1, col=1)

    fig_aot.add_trace(go.Scatter(x=exposure["Timestamp"], y=exposure["CumulativeExposure"], name="Cumulative",
                                 line=dict(color="purple")), row=2, col=1)

    fig_aot.update_layout(height=550)
    st.plotly_chart(fig_aot, use_container_width=True)

st.divider()

# 10) Vertical dispersion projection (hypothetical)
st.header("10) Hypothetical vertical dispersion profile")
if "PM2.5" in work and work["PM2.5"].notna().any():
    peak_pm25 = work["PM2.5"].quantile(0.9)
    C0 = float(peak_pm25)
    k = 0.05
    z_heights = np.arange(0, 100, 5)
    Cz = C0 * np.exp(-k * z_heights)

    fig_disp = go.Figure()
    fig_disp.add_trace(go.Scatter(x=Cz, y=z_heights, mode="lines+markers", line=dict(color="darkgreen")))
    fig_disp.add_hline(y=12, line_width=2, line_dash="dash", line_color="red")
    fig_disp.update_layout(
        xaxis_title="Estimated PM2.5 concentration (µg/m³)",
        yaxis_title="Height above ground (m)",
        title=f"Vertical dispersion (peak PM2.5 ~ {C0:.1f} µg/m³)"
    )
    st.plotly_chart(fig_disp, use_container_width=True)

st.divider()

# Insights and downloads
st.header("Summary and insights")

# Period stats
order = ["Pre-Diwali", "During Diwali", "Post-Diwali"]
if {"PM2.5", "PM10", "Period"} <= set(work.columns):
    period_stats = work.groupby("Period")[ ["PM2.5", "PM10"] ].agg(["mean", "median", "max"]).reindex(order)
    st.dataframe(period_stats, use_container_width=True)

st.markdown(
    """
    Key takeaways:
    - PM levels tend to rise around Diwali (During), then normalize Post-Diwali.
    - Temperature and humidity can modulate particulate concentrations.
    - Evening/night hours often exhibit higher PM due to atmospheric stability and festive activities.
    - AQI categories highlight the share of time spent in healthier vs unhealthy ranges.
    """
)

# Report Downloads & Documentation
st.header("📄 Reports & Documentation")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Export")
    # Enhanced CSV export with metadata
    export_metadata = f"""# Air Quality Analysis Export
# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
# Date Range: {work['Timestamp'].min().date()} to {work['Timestamp'].max().date()}
# Total Records: {len(work):,}
# Diwali Date: {DIWALI_DATE.date()}
# PM2.5 Threshold: {pm25_threshold} µg/m³
# Smoothing Window: {smooth_win} hours\n\n"""
    
    csv_with_metadata = export_metadata + work.to_csv(index=False)
    csv_bytes = csv_with_metadata.encode("utf-8")
    st.download_button(
        "📊 Download Filtered Data (CSV)", 
        data=csv_bytes, 
        file_name=f"air_quality_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", 
        mime="text/csv",
        help="Download current filtered dataset with analysis metadata"
    )

with col2:
    st.subheader("Analysis Report")
    # Check if PDF exists
    pdf_path = "PM_Analysis_Diwali2025_Final_Report.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(
            "📑 Download Full Analysis Report (PDF)",
            data=pdf_bytes,
            file_name="PM_Analysis_Diwali2025_Final_Report.pdf",
            mime="application/pdf",
            help="Comprehensive analysis report with detailed methodology and findings"
        )
        st.caption("📋 Complete technical analysis with statistical methods and conclusions")
    else:
        st.info("PDF report not available in current deployment")

with col3:
    st.subheader("Quick Insights")
    # Generate summary statistics
    period_summary = work.groupby('Period')[['PM2.5', 'PM10']].agg(['mean', 'median']).round(1)
    summary_text = f"""**Key Findings:**
    
🎇 **During Diwali Period:**
• PM2.5: {period_summary.loc['During Diwali', ('PM2.5', 'mean')]:.1f} µg/m³ (avg)
• PM10: {period_summary.loc['During Diwali', ('PM10', 'mean')]:.1f} µg/m³ (avg)

📈 **Pollution Increase:**
• {((period_summary.loc['During Diwali', ('PM2.5', 'mean')] / period_summary.loc['Pre-Diwali', ('PM2.5', 'mean')] - 1) * 100):.0f}% PM2.5 increase vs Pre-Diwali
• {((period_summary.loc['During Diwali', ('PM10', 'mean')] / period_summary.loc['Pre-Diwali', ('PM10', 'mean')] - 1) * 100):.0f}% PM10 increase vs Pre-Diwali

🌅 **Recovery:**
• Post-Diwali levels: {period_summary.loc['Post-Diwali', ('PM2.5', 'mean')]:.1f} µg/m³ PM2.5"""
    
    st.markdown(summary_text)

st.success("🎯 Dashboard ready. Use the sidebar controls to explore different time periods and thresholds.")
