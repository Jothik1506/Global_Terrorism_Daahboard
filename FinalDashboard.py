# Final Build Update
import os
...

import os
import warnings
import zipfile
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
#  PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="Terrorism Data Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
st.title("📊 Global Terrorism EDA Dashboard")
st.caption("Interactive dashboard analyzing global terrorism trends.")

st.markdown(
    """
    <style>
    div.block-container{padding-top:2rem;}
    .pill-row{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;align-items:center;text-align:center;}
    .pill{padding:4px 10px;border-radius:9999px;display:inline-block;}
    .pill.years{background:#eef2ff;color:#1e3a8a;}
    .pill.regions{background:#ecfeff;color:#155e75;}
    .pill.countries{background:#f0fdf4;color:#166534;}
    </style>
    """,
    unsafe_allow_html=True
)

px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Plasma"

# ------------------------------------------------------------
#  DATA PREPROCESSING
# ------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
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

    rename_dict = {}
    for col in df.columns:
        lower = col.lower()
        if lower in col_mapping:
            rename_dict[col] = col_mapping[lower]
        else:
            rename_dict[col] = col

    df = df.rename(columns=rename_dict)

    # Safe handling of Year column
    if 'Year' not in df.columns:
        df['Year'] = 'Unknown'
    
    # FORCE Year to be string to avoid sorting errors
    df['Year'] = df['Year'].fillna('Unknown').astype(str).str.replace('.0', '', regex=False)

    if 'Killed' not in df.columns:
        df['Killed'] = 0
    df['Killed'] = df['Killed'].fillna(0).astype(int)

    if 'Wounded' not in df.columns:
        df['Wounded'] = 0
    df['Wounded'] = df['Wounded'].fillna(0).astype(int)

    for field in ['Region', 'Country', 'Attack_Type', 'Target_Type', 'Group_Name']:
        if field not in df.columns:
            df[field] = 'Unknown'

    return df

# ------------------------------------------------------------
#  DATA LOADING (MEMORY OPTIMIZED)
# ------------------------------------------------------------
DATA_FILENAME = "dashboard_data.zip" 

@st.cache_data(show_spinner=True)
def load_dataset():
    """Load ONLY the necessary columns to save memory."""
    if not os.path.exists(DATA_FILENAME):
        st.error(f"❌ Error: The file '{DATA_FILENAME}' was not found.")
        st.info("Please ensure 'dashboard_data.zip' is uploaded to your GitHub repo.")
        st.stop()
        
    # CRITICAL: Only load these columns to prevent RAM crash
    cols_to_keep = [
        'iyear', 'country_txt', 'region_txt', 'attacktype1_txt', 
        'targtype1_txt', 'gname', 'nkill', 'nwound'
    ]

    try:
        # Handle ZIP file
        if DATA_FILENAME.endswith('.zip'):
            with zipfile.ZipFile(DATA_FILENAME, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f]
                if not csv_files:
                    st.error("❌ Error: No CSV file found inside the ZIP archive.")
                    st.stop()
                
                target_file = csv_files[0]
                with z.open(target_file) as f:
                    # usecols limits memory usage significantly
                    df_raw = pd.read_csv(f, encoding='latin1', usecols=cols_to_keep)
        
        # Handle standard CSV (if you unzipped it on the server)
        elif DATA_FILENAME.endswith('.csv'):
             df_raw = pd.read_csv(DATA_FILENAME, encoding='latin1', usecols=cols_to_keep)
        
        else:
            df_raw = pd.read_excel(DATA_FILENAME, usecols=cols_to_keep)
            
        df = preprocess_data(df_raw)
        return df

    except Exception as e:
        st.error(f"❌ Critical Error loading data: {e}")
        st.stop()

# ------------------------------------------------------------
#  EXECUTE LOAD
# ------------------------------------------------------------
st.sidebar.header("📂 Data Source")

# Load data
df = load_dataset()

if df is None:
    st.error("Failed to load data.")
    st.stop()

try:
    mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.sidebar.caption(f"RAM usage: {mem:.1f} MB")
except Exception:
    pass

# ------------------------------------------------------------
#  FILTER CONTROLS
# ------------------------------------------------------------
with st.sidebar.expander("🎛️ Filter Controls", expanded=True):
    all_years = sorted(df['Year'].unique().tolist())
    
    if not all_years:
        st.error("No Year data available.")
        st.stop()

    col_btn_a, col_btn_b = st.columns(2)
    with col_btn_a:
        if st.button("Reset", key="reset_filters"):
            st.session_state['start_year'] = all_years[0]
            st.session_state['end_year'] = all_years[-1]
            st.session_state['saved_regions'] = sorted(df['Region'].unique().tolist())
            st.session_state['saved_countries'] = sorted(df['Country'].unique().tolist())
            st.rerun()
    with col_btn_b:
        if st.button("Last 5 Years", key="last5"):
            last5 = all_years[-5:] if len(all_years) >= 5 else all_years
            st.session_state['start_year'] = last5[0]
            st.session_state['end_year'] = last5[-1]
            st.rerun()

    min_year = st.select_slider(
        "Start Year",
        options=all_years,
        value=st.session_state.get('start_year', all_years[0]),
        key="start_year"
    )
    max_year = st.select_slider(
        "End Year",
        options=all_years,
        value=st.session_state.get('end_year', all_years[-1]),
        key="end_year"
    )
    if all_years.index(min_year) > all_years.index(max_year):
        st.error("Start year cannot be after end year.")
        st.stop()

    region_options = sorted(df['Region'].unique())
    selected_regions = st.multiselect(
        "🌍 Regions",
        region_options,
        default=st.session_state.get('saved_regions', region_options),
        key="regions"
    )

    country_options = sorted(df[df['Region'].isin(selected_regions)]['Country'].unique())
    selected_countries = st.multiselect(
        "🏳️ Countries",
        country_options,
        default=st.session_state.get('saved_countries', country_options),
        key="countries"
    )

    search_term = st.text_input("🔍 Quick search (country/group)", key="search_box")
    if search_term:
        matches = [c for c in country_options if search_term.lower() in c.lower()]
        if matches:
            selected_countries = matches

    # Reduced sample threshold because of memory limits
    use_sample = st.checkbox("💡 Use sampled data for faster visuals", value=(len(df) > 100_000), key="sample_toggle")
    sample_rows = None
    if use_sample and len(df) > 5000:
        sample_rows = st.slider("Sample size", 5000, min(len(df), 100_000), 50000, step=5000)

df_active = df.copy()
if use_sample and sample_rows:
    df_active = df.sample(sample_rows, random_state=42)
    st.sidebar.caption(f"Using sampled subset: {len(df_active):,} rows")

df_filtered = df_active[
    (df_active['Year'] >= min_year) &
    (df_active['Year'] <= max_year) &
    (df_active['Region'].isin(selected_regions)) &
    (df_active['Country'].isin(selected_countries))
]

if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ------------------------------------------------------------
#  FILTER SUMMARY
# ------------------------------------------------------------
st.markdown("### Current Filters")
st.markdown(
    f"""
    <div class="pill-row">
        <span class="pill years">Years: {min_year} - {max_year}</span>
        <span class="pill regions">Regions: {', '.join(selected_regions) if selected_regions else 'All'}</span>
        <span class="pill countries">Countries: {len(selected_countries)} selected</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
#  KPIs
# ------------------------------------------------------------
st.markdown("## Key Statistics")
col1, col2, col3, col4 = st.columns(4)
total_attacks = len(df_filtered)
total_killed = df_filtered['Killed'].sum()
avg_fatalities = total_killed / total_attacks if total_attacks else 0
total_wounded = df_filtered['Wounded'].sum()

df_kpi_year = df_filtered.groupby('Year').agg(
    Attack_Count=('Year', 'size'),
    Total_Killed=('Killed', 'sum'),
    Total_Wounded=('Wounded', 'sum')
).reset_index().sort_values('Year')

yoy_attacks = yoy_killed = yoy_avg = yoy_wounded = None
if len(df_kpi_year) >= 2:
    last = df_kpi_year.iloc[-1]
    prev = df_kpi_year.iloc[-2]
    yoy_attacks = ((last['Attack_Count'] - prev['Attack_Count']) / prev['Attack_Count'] * 100) if prev['Attack_Count'] else None
    yoy_killed = ((last['Total_Killed'] - prev['Total_Killed']) / prev['Total_Killed'] * 100) if prev['Total_Killed'] else None
    prev_avg = (prev['Total_Killed'] / prev['Attack_Count']) if prev['Attack_Count'] else None
    curr_avg = (last['Total_Killed'] / last['Attack_Count']) if last['Attack_Count'] else None
    yoy_avg = ((curr_avg - prev_avg) / prev_avg * 100) if prev_avg else None
    yoy_wounded = ((last['Total_Wounded'] - prev['Total_Wounded']) / prev['Total_Wounded'] * 100) if prev['Total_Wounded'] else None

with col1:
    st.metric("Total Incidents", f"{total_attacks:,}", None if yoy_attacks is None else f"{yoy_attacks:.1f}%")
with col2:
    st.metric("Total Fatalities", f"{total_killed:,}", None if yoy_killed is None else f"{yoy_killed:.1f}%")
with col3:
    st.metric("Avg. Fatalities / Attack", f"{avg_fatalities:.2f}", None if yoy_avg is None else f"{yoy_avg:.1f}%")
with col4:
    st.metric("Total Wounded", f"{total_wounded:,}", None if yoy_wounded is None else f"{yoy_wounded:.1f}%")

st.markdown("---")

# ------------------------------------------------------------
#  VISUALIZATION TABS
# ------------------------------------------------------------
tab_geo, tab_time, tab_attack, tab_quality = st.tabs([
    "🌍 Geography", "📈 Time Trends", "🎯 Attacks & Groups", "🧮 Data Quality"
])

with tab_geo:
    st.subheader("World View")
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

    st.subheader("Fatalities by Region")
    region_stats = df_filtered.groupby('Region')['Killed'].sum().reset_index().sort_values('Killed', ascending=False)
    fig_region = px.bar(
        region_stats,
        x='Region',
        y='Killed',
        color='Region',
        title='Total Fatalities by Region'
    )
    st.plotly_chart(fig_region, use_container_width=True)

    st.subheader("Top 10 Countries by Incident Count")
    top_countries = df_filtered['Country'].value_counts().head(10).reset_index()
    top_countries.columns = ['Country', 'Count']
    fig_top = px.bar(
        top_countries,
        y='Country',
        x='Count',
        orientation='h',
        title='Top 10 Countries',
        color='Count'
    )
    fig_top.update_yaxes(autorange='reversed')
    st.plotly_chart(fig_top, use_container_width=True)

    pick_country = st.selectbox(
        "Quick view by country",
        options=["All"] + top_countries['Country'].tolist(),
        index=0
    )
    if pick_country != "All":
        st.info(f"Showing sample rows for {pick_country}")
        st.dataframe(df_filtered[df_filtered['Country'] == pick_country].head(20), use_container_width=True)
    
    # Creator credits
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 15px; margin-top: 30px; color: #666; font-size: 0.85em; border-top: 1px solid #e0e0e0;">
            <div><strong>This Dataset is Created by:</strong></div>
            <div>Vanam Jothik Krishna Siva Naga Sai Kanth</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab_time:
    st.subheader("Trend Analysis")
    df_time = df_filtered.groupby('Year').agg(
        Attacks=('Year', 'count'),
        Fatalities=('Killed', 'sum')
    ).reset_index()

    fig_attacks = px.line(df_time, x='Year', y='Attacks', markers=True, title='Incidents Over Time')
    fig_fatalities = px.line(df_time, x='Year', y='Fatalities', markers=True, color_discrete_sequence=['red'],
                             title='Fatalities Over Time')
    st.plotly_chart(fig_attacks, use_container_width=True)
    st.plotly_chart(fig_fatalities, use_container_width=True)

    df_time['MA3'] = pd.to_numeric(df_time['Attacks'], errors='coerce').rolling(3, min_periods=1).mean()
    fig_ma = px.line(df_time, x='Year', y=['Attacks', 'MA3'], title='3-Year Moving Average (Incidents)')
    st.plotly_chart(fig_ma, use_container_width=True)

    try:
        vals = pd.to_numeric(df_time['Attacks'], errors='coerce')
        z_scores = (vals - vals.mean()) / (vals.std() if vals.std() else 1)
        df_time['Z'] = z_scores
        anomalies = df_time[df_time['Z'].abs() >= 3]
        if not anomalies.empty:
            st.warning("Anomalies detected (|Z| ≥ 3).")
            st.dataframe(anomalies[['Year', 'Attacks', 'Z']], use_container_width=True)
    except Exception:
        pass

    st.download_button(
        "Download filtered dataset (CSV)",
        df_filtered.to_csv(index=False).encode('utf-8'),
        file_name="filtered_terrorism_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px; padding: 15px; margin-top: 30px; color: #666; font-size: 0.85em; border-top: 1px solid #e0e0e0;">
            <div><strong>This Dataset is Created by:</strong></div>
            <div>Vanam Jothik Krishna Siva Naga Sai Kanth</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab_attack:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Attack Type Distribution")
        attack_counts = df_filtered['Attack_Type'].value_counts().head(10).reset_index()
        attack_counts.columns = ['Attack_Type', 'Count']
        fig_attack = px.pie(
            attack_counts,
            values='Count',
            names='Attack_Type',
            title='Top Attack Types',
            hole=0.35
        )
        st.plotly_chart(fig_attack, use_container_width=True)

        st.subheader("Target Type Focus")
        target_counts = df_filtered['Target_Type'].value_counts().head(10).reset_index()
        target_counts.columns = ['Target_Type', 'Count']
        fig_target = px.bar(
            target_counts,
            x='Count',
            y='Target_Type',
            orientation='h',
            title='Most Common Targets'
        )
        fig_target.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_target, use_container_width=True)

    with col_right:
        st.subheader("Most Active Groups")
        group_df = df_filtered[df_filtered['Group_Name'].str.lower() != 'unknown']
        top_groups = group_df['Group_Name'].value_counts().head(10).reset_index()
        top_groups.columns = ['Group_Name', 'Count']
        fig_groups = px.bar(
            top_groups,
            y='Group_Name',
            x='Count',
            orientation='h',
            title='Top Terrorist Groups',
            color='Count'
        )
        fig_groups.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_groups, use_container_width=True)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px; padding: 15px; margin-top: 30px; color: #666; font-size: 0.85em; border-top: 1px solid #e0e0e0;">
            <div><strong>This Dataset is Created by:</strong></div>
            <div>Vanam Jothik Krishna Siva Naga Sai Kanth</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab_quality:
    st.subheader("Data Quality Checks")
    required_cols = ['Year', 'Region', 'Country', 'Killed', 'Wounded', 'Attack_Type', 'Target_Type', 'Group_Name']
    present_cols = [c for c in required_cols if c in df.columns]
    missing_cols = [c for c in required_cols if c not in df.columns]

    col_a, col_b = st.columns(2)
    with col_a:
        st.success(f"Present ({len(present_cols)}): " + ", ".join(present_cols) if present_cols else "Present (0)")
    with col_b:
        st.error(f"Missing ({len(missing_cols)}): " + ", ".join(missing_cols) if missing_cols else "Missing (0)")

    st.markdown("#### Unknown / Null Summary")
    quality_cols = ['Region', 'Country', 'Attack_Type', 'Target_Type', 'Group_Name']
    quality_stats = []
    for qc in quality_cols:
        if qc in df.columns:
            total_rows = len(df)
            unknown_cnt = (df[qc].astype(str).str.lower() == 'unknown').sum()
            null_cnt = df[qc].isna().sum()
            quality_stats.append({
                "Column": qc,
                "Unknown_Count": int(unknown_cnt),
                "Unknown_%": round(100 * unknown_cnt / total_rows, 2),
                "Null_Count": int(null_cnt),
                "Null_%": round(100 * null_cnt / total_rows, 2)
            })
    if quality_stats:
        st.dataframe(pd.DataFrame(quality_stats), use_container_width=True)
    else:
        st.info("No quality stats available.")

    st.markdown("#### Sample Problem Rows")
    for qc in quality_cols:
        if qc in df.columns:
            mask = df[qc].isna() | (df[qc].astype(str).str.lower() == 'unknown')
            if mask.any():
                st.caption(f"Issue column: {qc}")
                st.dataframe(
                    df.loc[mask, ['Year', 'Region', 'Country', 'Attack_Type', 'Target_Type', 'Group_Name', 'Killed', 'Wounded']].head(10),
                    use_container_width=True
                )
    st.success("Quality review complete.")
    
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px; padding: 15px; margin-top: 30px; color: #666; font-size: 0.85em; border-top: 1px solid #e0e0e0;">
            <div><strong>This Dataset is Created by:</strong></div>
            <div>Vanam Jothik Krishna Siva Naga Sai Kanth </div>        
        </div>
        """,
        unsafe_allow_html=True
    )