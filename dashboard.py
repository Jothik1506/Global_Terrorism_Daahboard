import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import os

warnings.filterwarnings("ignore")

# --- INITIAL SETUP ---
st.set_page_config(
    page_title="Terrorism Data Dashboard: EDA",
    page_icon=":bar_chart:",
    layout="wide"
)
st.title(":bar_chart: Global Terrorism EDA Dashboard")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)


# --- DATA PREPROCESSING FUNCTION ---
@st.cache_data
def preprocess_data(df):
    """Standardizes column names and cleans necessary fields."""
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

    # Create a mapping for current column names to their standardized versions (case-insensitive)
    rename_dict = {}
    found_iyear_col = False
    found_nkill_col = False
    found_nwound_col = False

    for original_col in df.columns:
        lower_col = original_col.lower()
        if lower_col in col_mapping:
            rename_dict[original_col] = col_mapping[lower_col]
            if lower_col == 'iyear':
                found_iyear_col = True
            elif lower_col == 'nkill':
                found_nkill_col = True
            elif lower_col == 'nwound':
                found_nwound_col = True
        else:
            rename_dict[original_col] = original_col # Keep original name if not in mapping

    # Rename columns first
    df.rename(columns=rename_dict, inplace=True)

    # Explicitly ensure 'Year' column exists and is of type string
    if not found_iyear_col and 'Year' not in df.columns:
        st.warning("The 'Year' column (or 'iyear') was not found in the dataset. Using 'Unknown' for year values.")
        df['Year'] = 'Unknown'
    # Now, it's safe to convert 'Year' to string, as it's guaranteed to exist
    df['Year'] = df['Year'].astype(str)

    # Explicitly ensure 'Killed' column exists and is of type int
    if not found_nkill_col and 'Killed' not in df.columns:
        st.warning("The 'Killed' column (or 'nkill') was not found. Using 0 for killed values.")
        df['Killed'] = 0
    df['Killed'] = df['Killed'].fillna(0).astype(int)

    # Explicitly ensure 'Wounded' column exists and is of type int
    if not found_nwound_col and 'Wounded' not in df.columns:
        # st.warning("The 'Wounded' column (or 'nwound') was not found. Using 0 for wounded values.")
        df['Wounded'] = 0 # Default to 0 if not present
    df['Wounded'] = df['Wounded'].fillna(0).astype(int) # Ensure it's int after fillna

    # Ensure other critical columns exist as 'Unknown' if not found
    for col_name in ['Region', 'Country', 'Attack_Type', 'Target_Type', 'Group_Name']:
        if col_name not in df.columns:
            # st.warning(f"The '{col_name}' column was not found. Using 'Unknown' for missing values.")
            df[col_name] = 'Unknown'

    return df


# --- FILE UPLOADER AND DATA LOADING ---

# Optional Data area (logo/branding)
_logo_path = r"/LOGO.jpg"
if os.path.exists(_logo_path):
    st.sidebar.image(_logo_path, use_container_width=True)
else:
    st.sidebar.info(f"Logo not found at '{_logo_path}'. Update the path or place the image accordingly.")

# The file path for the local dataset
LOCAL_FILE_NAME = r"D:\VS CODE\Python\dashboard_data.csv"

# The file uploader now accepts both CSV and XLSX
fl = st.file_uploader(":file_folder: Upload your terrorism dataset (.csv or .xlsx)", type=["csv", "xlsx"])
df = pd.DataFrame()  # Initialize empty DataFrame

if fl is not None:
    # 1. Load from uploaded file (Handles both CSV and XLSX)
    try:
        # Check file extension to use correct pandas reader
        if fl.name.lower().endswith('.csv'):
            df = pd.read_csv(fl, encoding='latin1')
        elif fl.name.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(fl)
        else:
            st.error("Unsupported file type uploaded.")
            st.stop()

        df = preprocess_data(df)
        st.success(f"Successfully loaded uploaded file: {fl.name}")
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        df = pd.DataFrame()  # Ensure df is empty on failure

else:
    # 2. Attempt to load the local file.
    st.info(
        f"No file uploaded. Attempting to load local file: `{LOCAL_FILE_NAME}`.")
    try:
        # Load the CSV file from the specified path
        df = pd.read_csv(LOCAL_FILE_NAME, encoding='latin1')
        df = preprocess_data(df)
        st.success("Successfully loaded data from local file.")
    except FileNotFoundError:
        st.warning(
            f"Local file not found: Please ensure the file exists at '{LOCAL_FILE_NAME}' OR use the uploader.")
    except Exception as e:
        st.error(f"Error processing local CSV file: {e}.")

if df.empty:
    st.warning("No data loaded. Please upload a file or ensure the local file path is correct.") # Added a more specific message
    st.stop()

# Show quick load summary
try:
    _mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    st.caption(f"Loaded {len(df):,} rows • ~{_mem_mb:.1f} MB")
except Exception:
    pass

st.dataframe(df.head(), use_container_width=True)

#//////////////// --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Options")

# Year Filter
all_years = sorted(df['Year'].unique().tolist())
# Quick controls
col_btn_1, col_btn_2 = st.sidebar.columns(2)
with col_btn_1:
    if st.button("Reset Filters", width='stretch'): # Changed use_container_width to width='stretch'
        st.session_state['start_year'] = all_years[0]
        st.session_state['end_year'] = all_years[-1]
        # Regions and countries will be (re)bound after widgets render
        st.session_state['regions'] = sorted(df['Region'].unique().tolist())
        st.session_state['countries'] = sorted(df['Country'].unique().tolist())
        st.rerun()
with col_btn_2:
    if st.button("Last 5 Years", width='stretch'): # Changed use_container_width to width='stretch'
        last5 = all_years[-5:] if len(all_years) >= 5 else all_years
        st.session_state['start_year'] = last5[0]
        st.session_state['end_year'] = last5[-1]
        st.rerun()

min_year = st.sidebar.select_slider(
    'Select Start Year',
    options=all_years,
    value=st.session_state.get('start_year', all_years[0]), # Use session state for persistence
    key='start_year'
)
max_year = st.sidebar.select_slider(
    'Select End Year',
    options=all_years,
    value=st.session_state.get('end_year', all_years[-1]), # Use session state for persistence
    key='end_year'
)
# Ensure start year is not after end year
if all_years.index(min_year) > all_years.index(max_year):
    st.sidebar.error("Start year cannot be after end year.")
    st.stop()

# Region Filter
region_list = sorted(df['Region'].unique().tolist())
selected_regions = st.sidebar.multiselect("Select Region(s)", region_list, default=st.session_state.get('regions', region_list), key='regions') # Use session state for persistence

# Country Filter (dynamically updated)
country_list = sorted(df[df['Region'].isin(selected_regions)]['Country'].unique().tolist())
selected_countries = st.sidebar.multiselect("Select Country/Countries", country_list, default=st.session_state.get('countries', country_list), key='countries') # Use session state for persistence

# Global quick search
search_query = st.sidebar.text_input("Search Country or Group (optional)", key="global_search").strip()
if search_query:
    matching_countries = [c for c in country_list if search_query.lower() in c.lower()]
    if matching_countries:
        selected_countries = matching_countries

# (Removed rate toggle and dependency on population data)

# --- FILTER DATASET ---
df_filtered = df[
    (df['Year'] >= min_year) &
    (df['Year'] <= max_year) &
    (df['Region'].isin(selected_regions)) &
    (df['Country'].isin(selected_countries))
    ]

# Check if filtered data is empty
if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- FILTER SUMMARY PILLS ---
st.markdown("### Current Filters")
st.markdown(
    f"""
    <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;align-items:center;text-align:center;">
      <span style="background:#eef2ff;color:#1e3a8a;padding:4px 8px;border-radius:9999px;display:inline-block;">Years: {min_year} - {max_year}</span>
      <span style="background:#ecfeff;color:#155e75;padding:4px 8px;border-radius:9999px;display:inline-block;">Regions: {', '.join(selected_regions) if selected_regions else 'All'}</span>
      <span style="background:#f0fdf4;color:#166534;padding:4px 8px;border-radius:9999px;display:inline-block;">Countries: {len(selected_countries)} selected</span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- KEY PERFORMANCE INDICATORS (KPIs) ---
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

# --- VISUALIZATIONS in TABS ---
st.markdown("## Data Visualizations")
tab1, tab2, tab3, tab4 = st.tabs(["Geographical Analysis", "Trend Analysis", "Attack & Group Analysis", "Data Quality"])

with tab1:
    st.subheader("Geographical Distribution of Incidents")

    # 1. World Map
    df_map = df_filtered.groupby('Country').size().reset_index(name='Incident_Count')
    _map_title = 'Total Incidents by Country'
    fig_map = px.choropleth(
        df_map,
        locations='Country',
        locationmode='country names',
        color='Incident_Count',
        hover_name='Country',
        color_continuous_scale=px.colors.sequential.Plasma,
        title=_map_title
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True) # unified

    # Optional sampling for visuals on large datasets
    if len(df_filtered) > 300000:
        st.info("Large dataset detected. You can sample data for faster visuals.")
        do_sample = st.checkbox("Sample data for visuals", value=True)
        sample_n = st.number_input("Rows to sample", min_value=10000, max_value=len(df_filtered), value=100000, step=10000)
        if do_sample:
            df_sampled = df_filtered.sample(int(sample_n), random_state=42) if len(df_filtered) > sample_n else df_filtered.copy()
        else:
            df_sampled = df_filtered.copy()
    else:
        df_sampled = df_filtered.copy()

    # 2. Total Killed by Region (Bar Chart)
    st.subheader("Total Fatalities by Region")
    df_killed_by_region = df_filtered.groupby('Region')['Killed'].sum().reset_index()
    df_killed_by_region = df_killed_by_region.sort_values(by='Killed', ascending=False)
    fig_region = px.bar(
        df_killed_by_region,
        x='Region',
        y='Killed',
        title='Fatalities by Region (Filtered)',
        template='plotly_white',
        color='Region'
    )
    st.plotly_chart(fig_region, use_container_width=True)

    # 3. Top 10 Countries by Incidents with quick filter
    st.subheader("Top 10 Countries by Incidents")
    country_counts = df_filtered.groupby('Country').size().reset_index(name='Count').sort_values('Count', ascending=False).head(10)
    # (Removed normalization per 100k)
    fig_top_countries = px.bar(
        country_counts,
        y='Country',
        x='Count',
        orientation='h',
        title='Top 10 Countries by Incident Count',
        template='plotly_white',
        labels={'Count': 'Number of Incidents', 'Country': 'Country'}
    )
    fig_top_countries.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_top_countries, use_container_width=True)

    # Country quick filter
    selected_top_country = st.selectbox("Filter to a Top Country (optional)", options=["All"] + country_counts['Country'].tolist(), index=0)
    if selected_top_country != "All":
        # Rerender summary table for the selected country
        st.info(f"Filtering view to: {selected_top_country}")
        df_country = df_filtered[df_filtered['Country'] == selected_top_country]
        st.dataframe(df_country.head(20), use_container_width=True)

with tab2:
    st.subheader("Analysis Over Time")

    # Prepare data for time series
    df_time_series = df_filtered.groupby('Year').agg(
        Attack_Count=('Year', 'size'),
        Total_Killed=('Killed', 'sum')
    ).reset_index()

    # 1. Attacks Over Time
    fig_time = px.line(
        df_time_series,
        x='Year',
        y='Attack_Count',
        title='Number of Incidents Per Year',
        markers=True,
        template='plotly_white',
        labels={'Attack_Count': 'Number of Attacks'}
    )
    fig_time.update_layout(xaxis={'tickmode': 'linear'})
    st.plotly_chart(fig_time, use_container_width=True)

    # 2. Fatalities Over Time
    fig_killed_time = px.line(
        df_time_series,
        x='Year',
        y='Total_Killed',
        title='Number of Fatalities Per Year',
        markers=True,
        template='plotly_white',
        labels={'Total_Killed': 'Number of Fatalities'},
        color_discrete_sequence=['red']
    )
    fig_killed_time.update_layout(xaxis={'tickmode': 'linear'})
    st.plotly_chart(fig_killed_time, use_container_width=True)

    # Simple forecast: 3-year moving average for incidents (visual aid)
    try:
        ts = df_time_series.copy()
        ts['Attack_Forecast_MA3'] = pd.to_numeric(ts['Attack_Count'], errors='coerce').rolling(3, min_periods=1).mean()
        fig_forecast = px.line(ts, x='Year', y=['Attack_Count', 'Attack_Forecast_MA3'],
                               title='Incidents with 3-year Moving Average', template='plotly_white')
        st.plotly_chart(fig_forecast, use_container_width=True)
    except Exception:
        pass

    # Anomaly detection (Z-score) on Attack_Count
    try:
        ts2 = df_time_series.copy()
        vals = pd.to_numeric(ts2['Attack_Count'], errors='coerce')
        z = (vals - vals.mean()) / (vals.std() if vals.std() else 1)
        ts2['Z'] = z
        anomalies = ts2[ts2['Z'].abs() >= 3]
        if not anomalies.empty:
            st.warning("Detected anomalies (|Z| >= 3) in yearly incidents:")
            st.dataframe(anomalies[['Year', 'Attack_Count', 'Z']], use_container_width=True)
    except Exception:
        pass

    # Download buttons for aggregated data
    st.markdown("### Downloads")
    st.download_button(
        "Download filtered dataset (CSV)",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name="filtered_terrorism_data.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download time series (CSV)",
        data=df_time_series.to_csv(index=False).encode('utf-8'),
        file_name="time_series.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download country incidents (CSV)",
        data=df_map.to_csv(index=False).encode('utf-8'),
        file_name="country_incidents.csv",
        mime="text/csv"
    )
    # Optional: download map image (requires kaleido)
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(fig_map, format="png", scale=2)
        st.download_button("Download map image (PNG)", data=img_bytes, file_name="map.png", mime="image/png")
    except Exception:
        st.caption("Install 'kaleido' to enable image downloads: pip install -U kaleido")

with tab3:
    st.subheader("Attack & Perpetrator Analysis")
    col_bottom_1, col_bottom_2 = st.columns(2)

    with col_bottom_1:
        # 3. Top 10 Attack Types
        st.subheader("Top 10 Attack Types")
        attack_type_counts = df_filtered.groupby('Attack_Type').size().reset_index(name='Count')
        attack_type_counts = attack_type_counts.sort_values(by='Count', ascending=False).head(10)
        fig_attack = px.pie(
            attack_type_counts,
            values='Count',
            names='Attack_Type',
            title='Distribution of Attack Types',
            hole=0.3
        )
        st.plotly_chart(fig_attack, use_container_width=True)

    with col_bottom_2:
        # 4. Top 10 Active Terrorist Groups
        # Group_Name is now guaranteed to exist due to preprocess_data improvements
        st.subheader("Top 10 Most Active Groups")
        df_groups = df_filtered[df_filtered['Group_Name'].str.lower() != 'unknown']
        group_counts = df_groups.groupby('Group_Name').size().reset_index(name='Count')
        group_counts = group_counts.sort_values(by='Count', ascending=False).head(10)
        fig_groups = px.bar(
            group_counts,
            y='Group_Name',
            x='Count',
            orientation='h',
            title='Groups by Incident Count',
            template='plotly_white',
            labels={'Group_Name': 'Terrorist Group', 'Count': 'Number of Incidents'}
        )
        fig_groups.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_groups, use_container_width=True)

with tab4:
    st.subheader("Data Quality & Validation")

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