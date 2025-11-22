# ============================================================
# 🧠 Global Terrorism EDA Dashboard (Improved Streamlit v2.0)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
#  PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="Global Terrorism EDA Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
st.title("📊 Global Terrorism Exploratory Dashboard")
st.caption("Analyze terrorism trends, regions, and groups interactively.")

st.markdown(
    """
    <style>
    div.block-container{padding-top:2rem;}
    .small-font {font-size:13px; color:gray;}
    </style>
    """, unsafe_allow_html=True
)

px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Plasma"

# ------------------------------------------------------------
#  DATA PREPROCESSING FUNCTION
# ------------------------------------------------------------
@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize terrorism dataset."""
    col_mapping = {
        'iyear': 'Year',
        'country_txt': 'Country',
        'region_txt': 'Region',
        'attacktype1_txt': 'Attack_Type',
        'targtype1_txt': 'Target_Type',
        'gname': 'Group_Name',
        'nkill': 'Killed',
        'nwound': 'Wounded'
    }

    # Normalize column names
    rename_dict = {col: col_mapping.get(col.lower(), col) for col in df.columns}
    df.rename(columns=rename_dict, inplace=True)

    # Fill required columns if missing
    for col in ['Year', 'Country', 'Region', 'Attack_Type', 'Target_Type', 'Group_Name']:
        if col not in df.columns:
            df[col] = 'Unknown'

    for col in ['Killed', 'Wounded']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    df['Year'] = df['Year'].astype(str)
    return df


# ------------------------------------------------------------
#  DATA LOADING
# ------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Globe_icon.svg", width=100)
st.sidebar.title("⚙️ Data & Filters")

LOCAL_FILE = r"C:\Users\Jothik\Downloads\globalterrorismdb_0718dist.csv"

file = st.sidebar.file_uploader("📂 Upload Dataset (.csv or .xlsx)", type=["csv", "xlsx"])
with st.spinner("Loading data..."):
    try:
        if file is not None:
            df = pd.read_csv(file, encoding='latin1') if file.name.endswith('.csv') else pd.read_excel(file)
            source = f"Uploaded file: {file.name}"
        else:
            df = pd.read_csv(LOCAL_FILE, encoding='latin1')
            source = f"Local file: {os.path.basename(LOCAL_FILE)}"
        df = preprocess_data(df)
        st.success(f"✅ Successfully loaded {source} ({len(df):,} rows)")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

st.caption(f"Data Memory: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
st.dataframe(df.head(), use_container_width=True)

# ------------------------------------------------------------
#  FILTERS
# ------------------------------------------------------------
with st.sidebar.expander("🎛️ Filter Controls", expanded=True):
    all_years = sorted(df['Year'].unique().tolist())
    min_year = st.select_slider("Start Year", options=all_years, value=all_years[0])
    max_year = st.select_slider("End Year", options=all_years, value=all_years[-1])
    if all_years.index(min_year) > all_years.index(max_year):
        st.error("⚠️ Start year cannot be after end year.")
        st.stop()

    region_list = sorted(df['Region'].unique())
    selected_regions = st.multiselect("🌍 Select Region(s)", region_list, default=region_list)

    country_list = sorted(df[df['Region'].isin(selected_regions)]['Country'].unique())
    selected_countries = st.multiselect("🏳️ Select Countries", country_list, default=country_list)

    search = st.text_input("🔍 Search Country or Group", help="Type part of a country or group name.")
    if search:
        selected_countries = [c for c in country_list if search.lower() in c.lower()]

    sample_toggle = st.checkbox("💡 Use sampled data (faster on large datasets)", value=(len(df) > 300000))
    if sample_toggle:
        n = st.slider("Sample size", 10000, min(300000, len(df)), 100000, step=10000)
        df = df.sample(n, random_state=42)
        st.sidebar.caption(f"Using {n:,} sampled rows.")

# Filter data
df_filtered = df[
    (df['Year'] >= min_year) & (df['Year'] <= max_year) &
    (df['Region'].isin(selected_regions)) &
    (df['Country'].isin(selected_countries))
]

if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()
#----------------------------------------------------------
# --- FILTER SUMMARY PILLS ---
#----------------------------------------------------------
st.markdown("### Current Filters")
st.markdown(
    f"""
    <div style="display:flex;flex-wrap:wrap;gap:8px;">
      <span style="background:#eef2ff;color:#1e3a8a;padding:4px 8px;border-radius:9999px;">Years: {min_year} - {max_year}</span>
      <span style="background:#ecfeff;color:#155e75;padding:4px 8px;border-radius:9999px;">Regions: {', '.join(selected_regions) if selected_regions else 'All'}</span>
      <span style="background:#f0fdf4;color:#166534;padding:4px 8px;border-radius:9999px;">Countries: {len(selected_countries)} selected</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
#  KPI METRICS
# ------------------------------------------------------------
st.markdown("## Key Statistics")
col1, col2, col3, col4 = st.columns(4)
total_attacks = len(df_filtered)
total_killed = df_filtered['Killed'].sum()
avg_fatalities_per_attack = total_killed / total_attacks if total_attacks > 0 else 0
total_wounded = df_filtered['Wounded'].sum() if 'Wounded' in df_filtered.columns else 0

# Compute YoY deltas for KPIs
df_kpi_by_year = df_filtered.groupby('Year').agg(
    Attack_Count=('Year', 'size'),
    Total_Killed=('Killed', 'sum'),
    Total_Wounded=('Wounded', 'sum') # Wounded is guaranteed to exist now
).reset_index().sort_values('Year')

# Ensure we have at least 2 years for deltas
yoy_attack_delta = None
yoy_killed_delta = None
yoy_avg_fatal_delta = None
yoy_wounded_delta = None
if len(df_kpi_by_year) >= 2:
    last = df_kpi_by_year.iloc[-1]
    prev = df_kpi_by_year.iloc[-2]
    # Avoid division by zero
    yoy_attack_delta = ((last['Attack_Count'] - prev['Attack_Count']) / prev['Attack_Count'] * 100) if prev['Attack_Count'] else None
    yoy_killed_delta = ((last['Total_Killed'] - prev['Total_Killed']) / prev['Total_Killed'] * 100) if prev['Total_Killed'] else None
    prev_avg = (prev['Total_Killed'] / prev['Attack_Count']) if prev['Attack_Count'] else None
    curr_avg = (last['Total_Killed'] / last['Attack_Count']) if last['Attack_Count'] else None
    yoy_avg_fatal_delta = ((curr_avg - prev_avg) / prev_avg * 100) if prev_avg else None
    yoy_wounded_delta = ((last['Total_Wounded'] - prev['Total_Wounded']) / prev['Total_Wounded'] * 100) if prev['Total_Wounded'] else None

with col1:
    st.metric("Total Terrorist Incidents", f"{total_attacks:,}", None if yoy_attack_delta is None else f"{yoy_attack_delta:.1f}%")
with col2:
    st.metric("Total Fatalities", f"{total_killed:,}", None if yoy_killed_delta is None else f"{yoy_killed_delta:.1f}%")
with col3:
    st.metric("Avg. Fatalities Per Attack", f"{avg_fatalities_per_attack:.2f}", None if yoy_avg_fatal_delta is None else f"{yoy_avg_fatal_delta:.1f}%")
with col4:
    st.metric("Total Wounded", f"{total_wounded:,}", None if yoy_wounded_delta is None else f"{yoy_wounded_delta:.1f}%")

st.markdown("---")
# ------------------------------------------------------------
#  VISUALIZATION TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Geography", "📈 Time Trends", "🎯 Attacks & Groups", "🧮 Data Quality"
])

# ============================================================
# TAB 1: GEOGRAPHY
# ============================================================
with tab1:
    st.subheader("🌍 Geographical Distribution")

    df_map = df_filtered.groupby('Country').size().reset_index(name='Incident_Count')
    fig_map = px.choropleth(
        df_map,
        locations='Country',
        locationmode='country names',
        color='Incident_Count',
        hover_name='Country',
        title='Incidents by Country'
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    df_killed_by_region = df_filtered.groupby('Region')['Killed'].sum().reset_index()
    fig_region = px.bar(df_killed_by_region, x='Region', y='Killed', color='Region',
                        title='Total Fatalities by Region', template='plotly_white')
    st.plotly_chart(fig_region, use_container_width=True)

    st.subheader("Top 10 Countries by Incidents")
    top_countries = df_filtered['Country'].value_counts().head(10).reset_index()
    top_countries.columns = ['Country', 'Count']
    fig_top = px.bar(top_countries, y='Country', x='Count', orientation='h',
                     title='Top 10 Countries', color='Count')
    fig_top.update_yaxes(autorange='reversed')
    st.plotly_chart(fig_top, use_container_width=True)

    # Country quick filter
    st.subheader("Top Country Filter")

    country_list = df_filtered['Country'].value_counts().index.tolist()

    selected_country = st.selectbox(
        "Filter to a Top Country (optional)",
        options=["All"] + country_list,
        index=0
    )

    if selected_country != "All":
        st.info(f"Filtering view to: {selected_country}")
        df_country = df_filtered[df_filtered['Country'] == selected_country]
        st.dataframe(df_country.head(20), use_container_width=True)

# ============================================================
# TAB 2: TIME TRENDS
# ============================================================
with tab2:
    st.subheader("📈 Time Trend Analysis")
    df_time = df_filtered.groupby('Year').agg(Attacks=('Year', 'count'),
                                              Killed=('Killed', 'sum')).reset_index()

    fig_attacks = px.line(df_time, x='Year', y='Attacks', markers=True,
                          title='Number of Attacks Over Time')
    fig_killed = px.line(df_time, x='Year', y='Killed', markers=True,
                         title='Fatalities Over Time', color_discrete_sequence=['red'])

    st.plotly_chart(fig_attacks, use_container_width=True)
    st.plotly_chart(fig_killed, use_container_width=True)

    # Simple moving average
    df_time['MA3'] = df_time['Attacks'].rolling(3, min_periods=1).mean()
    fig_ma = px.line(df_time, x='Year', y=['Attacks', 'MA3'],
                     title='3-Year Moving Average (Incidents)', labels={'value': 'Attacks'})
    st.plotly_chart(fig_ma, use_container_width=True)

    # --- Anomaly detection (Z-score) on Attack_Count ---
    ts2 = df_time.copy()
    ts2['Attack_Count'] = ts2['Attacks']  # rename for consistency

    vals = pd.to_numeric(ts2['Attack_Count'], errors='coerce')
    z = (vals - vals.mean()) / (vals.std() if vals.std() else 1)
    ts2['Z'] = z

    anomalies = ts2[ts2['Z'].abs() >= 3]

    if not anomalies.empty:
        st.warning("Detected anomalies (|Z| >= 3) in yearly incidents:")
        st.dataframe(anomalies[['Year', 'Attack_Count', 'Z']], use_container_width=True)

    # Download
    st.download_button("📥 Download Filtered Data (CSV)",
                       df_filtered.to_csv(index=False).encode('utf-8'),
                       file_name="filtered_terrorism.csv", mime="text/csv")

# ============================================================
# TAB 3: ATTACKS & GROUPS
# ============================================================
with tab3:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Top 10 Attack Types")
        attack_types = df_filtered['Attack_Type'].value_counts().head(10).reset_index()
        attack_types.columns = ['Attack_Type', 'Count']
        fig_attack = px.pie(attack_types, values='Count', names='Attack_Type',
                            title='Attack Type Distribution', hole=0.3)
        st.plotly_chart(fig_attack, use_container_width=True)

        st.subheader("Top 10 Target Types")
        targets = df_filtered['Target_Type'].value_counts().head(10).reset_index()
        targets.columns = ['Target_Type', 'Count']
        fig_target = px.bar(targets, x='Count', y='Target_Type', orientation='h',
                            title='Most Common Target Types')
        fig_target.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_target, use_container_width=True)

    with colB:
        st.subheader("Top 10 Active Terrorist Groups")
        groups = df_filtered[df_filtered['Group_Name'].str.lower() != 'unknown']
        top_groups = groups['Group_Name'].value_counts().head(10).reset_index()
        top_groups.columns = ['Group_Name', 'Count']
        fig_groups = px.bar(top_groups, y='Group_Name', x='Count', orientation='h',
                            title='Most Active Groups', color='Count')
        fig_groups.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_groups, use_container_width=True)

# ============================================================
# TAB 4: DATA QUALITY
# ============================================================
with tab4:
        # Column checklist
    required_cols = ['Year', 'Region', 'Country', 'Killed', 'Wounded', 'Attack_Type', 'Target_Type', 'Group_Name']
    present_cols = [col for col in required_cols if col in df.columns]
    missing_cols = [col for col in required_cols if col not in df.columns]

    st.markdown("#### Column Checklist")
    col_a, col_b = st.columns(2)
    with col_a:
        st.success(f"Present ({len(present_cols)}): " + ", ".join(present_cols) if present_cols else "Present (0)")
    with col_b:
        st.error(f"Missing ({len(missing_cols)}): " + ", ".join(missing_cols) if missing_cols else "Missing (0)")

    # Unknown and null rates
    st.markdown("#### Unknown/Null Rates")
    quality_cols = ['Region', 'Country', 'Attack_Type', 'Target_Type', 'Group_Name']
    stats_rows = []
    for qc in quality_cols:
        if qc in df.columns:
            total = len(df)
            unknown = (df[qc].astype(str).str.lower() == 'unknown').sum()
            nulls = df[qc].isna().sum()
            stats_rows.append({
                "Column": qc,
                "Unknown_Count": int(unknown),
                "Unknown_%": round(100 * unknown / total, 2),
                "Null_Count": int(nulls),
                "Null_%": round(100 * nulls / total, 2),
            })
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
    else:
        st.info("No quality columns found to summarize.")

    # Problem samples
    st.markdown("#### Sample Problem Rows")
    issues = []
    for qc in quality_cols:
        if qc in df.columns:
            mask = df[qc].isna() | (df[qc].astype(str).str.lower() == 'unknown')
            if mask.any():
                sample = df.loc[mask, ['Year', 'Region', 'Country', 'Attack_Type', 'Target_Type', 'Group_Name', 'Killed', 'Wounded']].head(10)
                issues.append((qc, sample))
    if issues:
        for name, sample_df in issues:
            st.caption(f"Issue column: {name}")
            st.dataframe(sample_df, use_container_width=True)
    else:
        st.success("No rows with Unknown/Null in key columns.")
