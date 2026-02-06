import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
import os
import requests
from typing import Optional
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Global Styles - Wide landscape layout with new color scheme
style_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Montserrat:wght@400;600;700&display=swap');
:root{--accent:#6C6E36;--neutral:#C0BDBC;--background:#DFD3B5;--text:#3B3B3B}
html{scroll-behavior:smooth}
body, .stApp {font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color:#3B3B3B; background:#DFD3B5}
h1,h2,h3{font-family:'Montserrat', 'Inter', sans-serif;margin:0;color:#6C6E36}
h1{font-size:40px;line-height:1.1;margin-bottom:8px}
h2{font-size:20px;margin-bottom:6px}

/* Wide content layout */
.main .block-container {
    max-width: 95% !important;
    padding-left: 2.5% !important;
    padding-right: 2.5% !important;
    padding-top: 0.2rem !important;
}

/* Buttons */
.stButton>button{background:#6C6E36;color:white;border-radius:6px;padding:8px 12px;border:0;font-weight:600}
.stButton>button:hover{filter:brightness(1.1);transform:translateY(-1px)}

/* Reduce vertical spacing */
.element-container {margin-bottom: 0.5rem;}
</style>
"""

st.markdown(style_html, unsafe_allow_html=True)

# White header bar behind logo
# st.markdown("""
# <div style="
#     position: fixed;
#     top: 0;
#     left: 0;
#     right: 0;
#     height: 177px;
#     background: #F9F3EA;
#     z-index: 0;
#     box-shadow: 0 2px 4px rgba(0,0,0,0.1);
# "></div>
# """, unsafe_allow_html=True)

# Add space at top to drop image down
st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

# Full-width background bar behind navbar
st.markdown("""
<div style="
    margin-left: calc(-50vw + 50%);
    margin-right: calc(-50vw + 50%);
    width: 100vw;
    height: 160px;
    background: #F9F3EA;
    position: relative;
    margin-top: -28px;
    margin-bottom: -160px;
    z-index: 0;
"></div>
""", unsafe_allow_html=True)

# Navbar Image - Centered at larger width, positioned over background bar
navbar_candidates = [
    'images/new_nav_bar.png',
    'images/new_nav_bar.jpg',
    'new_nav_bar.png',
    'new_nav_bar.jpg'
]

# Display navbar image centered at larger width
navbar_found = False
for navbar_path in navbar_candidates:
    if os.path.exists(navbar_path):
        # Create three columns to center the image (optimal size for resolution)
        left_spacer, img_col, right_spacer = st.columns([1, 1.5, 1])
        with img_col:
            st.image(navbar_path, use_column_width=True)
        navbar_found = True
        break

if not navbar_found:
    st.warning("Navbar image 'new_nav_bar.png' not found. Please add it to the images/ folder.")

# Tighten space between image and content
st.markdown("<div style='margin-bottom: -50px;'></div>", unsafe_allow_html=True)

# ============================================================================
# SPECIES NAME MAPPING
# ============================================================================

def get_species_full_name(code):
    """Convert species code to full name."""
    if not code or pd.isna(code):
        return "Unknown"

    # Clean the code (remove whitespace, citations, etc.)
    code = str(code).strip().upper()
    # Remove [CITE: xxx] or [cite: xxx] patterns
    code = code.split('[')[0].strip()
    # Remove trailing commas and slashes
    code = code.split(',')[0].split('/')[0].strip()

    # Species mapping dictionary
    species_map = {
        # Ducks
        'MALL': 'Mallard',
        'NOPI': 'Northern Pintail',
        'PNTL': 'Northern Pintail',
        'GADW': 'Gadwall',
        'AGWT': 'Green-winged Teal',
        'AMWI': 'American Wigeon',
        'NSHO': 'Northern Shoveler',
        'NOSHI': 'Northern Shoveler',
        'BUFF': 'Bufflehead',
        'WDDU': 'Wood Duck',
        'WODU': 'Wood Duck',
        'CINT': 'Cinnamon Teal',
        'BWTE': 'Blue-winged Teal',
        'CANV': 'Canvasback',
        'REDH': 'Redhead',
        'RING': 'Ring-necked Duck',
        'RNDU': 'Ring-necked Duck',
        'RUDU': 'Ruddy Duck',
        'SCAU': 'Scaup',
        'GOLD': 'Goldeneye',
        'OTDU': 'Other Duck',
        # Geese
        'GWFG': 'White-fronted Goose',
        'CAGO': 'Canada Goose',
        'CACG': 'Cackling Goose',
        'ACGO': 'Aleutian Goose',
        'LSGO': 'Snow Goose',
        'ROGO': 'Ross\'s Goose',
        'OTGO': 'Other Goose',
        # Other
        'NONE': 'None',
        'CITE': 'Unknown'
    }

    return species_map.get(code, code)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_all_harvest_data():
    """
    Automatically find and load all harvest CSV files in the directory.
    Combines them into a single master DataFrame with standardized dates.
    """
    import os
    import re
    
    directory = '.'
    all_files = os.listdir(directory)
    
    # Find all harvest CSV files (matching pattern like "06-07 Harvest F.F.csv" or "24-25 harvest F.F.csv")
    harvest_files = []
    for file in all_files:
        if file.endswith('.csv'):
            # Check if it contains harvest data (skip Historical_Baseline, Refuge_Coordinates, etc.)
            if 'harvest' in file.lower() and 'combined' not in file.lower() and 'baseline' not in file.lower():
                harvest_files.append(file)
    
    # Sort files by year range
    def extract_year_range(filename):
        match = re.search(r'(\d{2})-(\d{2})', filename)
        if match:
            return int(match.group(1))
        return 0
    
    harvest_files.sort(key=extract_year_range)
    
    combined_df = pd.DataFrame()
    
    for file in harvest_files:
        try:
            filepath = os.path.join(directory, file)
            
            # Extract year range from filename
            match = re.search(r'(\d{2})-(\d{2})', file)
            if match:
                start_year = 2000 + int(match.group(1))
                end_year = 2000 + int(match.group(2))
            else:
                continue
            
            # Detect header row
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                header = 0 if 'Area Name' in first_line else 1
            
            # Read CSV with empty-file handling
            try:
                df = pd.read_csv(filepath, header=header, on_bad_lines='skip')
            except pd.errors.EmptyDataError:
                st.warning(f"Skipping empty CSV file: {file}")
                continue
            
            # Ensure required columns exist
            if 'Date' not in df.columns or 'Area Name' not in df.columns:
                st.warning(f"Skipping {file} - missing required columns ('Date' or 'Area Name')")
                continue
            
            # Standardize and parse dates
            def parse_date(date_str):
                if pd.isna(date_str) or not isinstance(date_str, str):
                    return pd.NaT
                try:
                    month, day = map(int, date_str.split('/'))
                    # Season runs Oct-Dec, so Oct-Dec = current year, Jan-Sep = next year
                    year = start_year if month >= 10 else end_year
                    return pd.Timestamp(year=year, month=month, day=day)
                except:
                    return pd.NaT
            
            df['Date'] = df['Date'].apply(parse_date)
            
            # Calculate Season_Week
            season_start = pd.Timestamp(year=start_year, month=10, day=1)
            df['Season_Week'] = ((df['Date'] - season_start).dt.days // 7) + 1
            
            # Standardize Area Name
            df['Area Name'] = df['Area Name'].str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Standardize species columns
            species_cols = [col for col in df.columns if 'Species' in col]
            for col in species_cols:
                if col in df.columns:
                    df[col] = df[col].str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Append to combined
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            
        except Exception as e:
            st.warning(f"Could not load {file}: {str(e)}")
            continue
    
    # Drop rows with invalid dates or missing required columns
    combined_df = combined_df.dropna(subset=['Date', 'Area Name'])
    
    return combined_df

def compute_historical_baseline(master_df):
    """
    Compute a historical baseline DataFrame from the aggregated master data if the precomputed baseline is missing.
    """
    results = []
    if master_df is None or master_df.empty:
        return pd.DataFrame(columns=['Area Name','Season_Week','15yr_Avg_Ducks','Top_Species','Prob_Successful_Hunt'])

    for (area, week), group in master_df.groupby(['Area Name', 'Season_Week']):
        # Historical average of Average Ducks
        avg_ducks = group['Average Ducks'].mean() if 'Average Ducks' in group.columns else float('nan')

        # Determine top species (use '#1 Species' or fallback '#1 Duck Species')
        if '#1 Species' in group.columns:
            species_series = group['#1 Species'].fillna(pd.NA)
        elif '#1 Duck Species' in group.columns:
            species_series = group['#1 Duck Species'].fillna(pd.NA)
        else:
            species_series = pd.Series([pd.NA] * len(group))

        top_species = species_series.mode().iloc[0] if not species_series.mode().empty else None

        # Probability of successful hunt: proportion of years with above-average yearly averages
        if 'Date' in group.columns and 'Average Ducks' in group.columns:
            yearly_avg = group.groupby(group['Date'].dt.year)['Average Ducks'].mean()
            overall_mean = yearly_avg.mean() if not yearly_avg.empty else float('nan')
            prob_success = float((yearly_avg > overall_mean).mean()) if not yearly_avg.empty else float('nan')
        else:
            prob_success = float('nan')

        results.append({
            'Area Name': area,
            'Season_Week': int(week),
            '15yr_Avg_Ducks': avg_ducks,
            'Top_Species': top_species,
            'Prob_Successful_Hunt': prob_success
        })

    return pd.DataFrame(results)

@st.cache_data(ttl=24*3600)
def fetch_weather_for_date(lat, lon, date):
    """Fetch daily weather (temp max/min, precipitation, windspeed) for a single date using Open-Meteo.
    Returns None if no data available or on error.
    """
    try:
        if pd.isna(lat) or pd.isna(lon):
            return None
        date = pd.to_datetime(date)
        date_str = date.strftime('%Y-%m-%d')
        today = pd.Timestamp.today().normalize()
        params = "daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode&timezone=auto"
        # Use forecast endpoint for today/future, archive for past
        if date >= today:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&{params}"
        else:
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&{params}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get('daily', {})
        times = daily.get('time', [])
        if not times:
            return None
        return {
            'date': pd.to_datetime(times[0]),
            'temp_max': daily.get('temperature_2m_max', [None])[0],
            'temp_min': daily.get('temperature_2m_min', [None])[0],
            'precipitation': daily.get('precipitation_sum', [None])[0],
            'windspeed': daily.get('windspeed_10m_max', [None])[0],
            'weathercode': daily.get('weathercode', [None])[0]
        }
    except Exception:
        return None

@st.cache_data(ttl=24*3600)
def analyze_weather_probability(area_name, date, years_back, precip_thresh_mm, wind_thresh_kmh, temp_min_c, temp_max_c, master_data_df, historical_df, coordinates_df):
    """Fetch weather for the chosen date across the past `years_back` years for the specified area,
    determine which years had 'good' weather according to thresholds, then compute the probability
    of a successful hunt given those weather conditions by comparing to historical harvest data.

    Returns a dict with the current weather, a DataFrame of historical entries, and probability metrics.
    """
    # Lookup coordinates
    row = coordinates_df[coordinates_df['Area Name'].str.upper() == area_name.upper()]
    if row.empty:
        return {'error': f'Coordinates not found for {area_name}'}
    lat = row.iloc[0]['Latitude']
    lon = row.iloc[0]['Longitude']
    date = pd.to_datetime(date)

    current = fetch_weather_for_date(lat, lon, date)

    records = []
    for y in range(date.year - years_back, date.year):
        try:
            d = date.replace(year=y)
        except Exception:
            # handle Feb 29 -> fallback to Feb 28
            d = pd.Timestamp(year=y, month=2, day=28)
        w = fetch_weather_for_date(lat, lon, d)

        # Compute whether that year had a 'successful' hunt based on master_data and historical baseline
        season_week = calculate_season_week(d)
        loc_week = master_data_df[
            (master_data_df['Area Name'].str.upper() == area_name.upper()) &
            (master_data_df['Season_Week'] == season_week) &
            (master_data_df['Date'].dt.year == y)
        ]
        success = False
        if not loc_week.empty and 'Average Ducks' in loc_week.columns:
            year_avg = loc_week['Average Ducks'].mean()
            # try to get baseline for that area/week
            hb_row = historical_df[
                (historical_df['Area Name'].str.upper() == area_name.upper()) &
                (historical_df['Season_Week'] == season_week)
            ]
            if not hb_row.empty:
                hb = hb_row['15yr_Avg_Ducks'].iloc[0]
                if not np.isnan(hb):
                    success = year_avg > hb
            else:
                # fallback: compare to median for the area/week across master data
                fallback_median = master_data_df[
                    (master_data_df['Area Name'].str.upper() == area_name.upper()) &
                    (master_data_df['Season_Week'] == season_week)
                ]['Average Ducks'].median()
                if not np.isnan(fallback_median):
                    success = year_avg > fallback_median

        weather_good = False
        if w:
            # convert wind to km/h (Open-Meteo reports in m/s), tolerant to None
            ws = w.get('windspeed', None)
            ws_kmh = ws * 3.6 if ws is not None else None
            precip = w.get('precipitation', None)
            tmin = w.get('temp_min', None)
            tmax = w.get('temp_max', None)
            if precip is not None and ws_kmh is not None and tmin is not None and tmax is not None:
                weather_good = (
                    (precip <= precip_thresh_mm) and
                    (ws_kmh <= wind_thresh_kmh) and
                    (tmin >= temp_min_c) and
                    (tmax <= temp_max_c)
                )
        records.append({
            'year': y,
            'date': d.date(),
            'temp_min': None if not w else w['temp_min'],
            'temp_max': None if not w else w['temp_max'],
            'precipitation_mm': None if not w else w['precipitation'],
            'windspeed_m_s': None if not w else w['windspeed'],
            'windspeed_kmh': None if not w or w['windspeed'] is None else w['windspeed'] * 3.6,
            'weather_good': weather_good,
            'successful_hunt': success
        })

    df = pd.DataFrame(records)
    n_good = int(df['weather_good'].sum()) if not df.empty else 0
    n_good_success = int(df[(df['weather_good']) & (df['successful_hunt'])].shape[0]) if not df.empty else 0
    prob_good = float(n_good_success / n_good) if n_good > 0 else None

    # Similarity to current weather (simple tolerances)
    similar_tol_temp = 2.0
    similar_tol_wind = 5.0
    if current is not None and not df.empty:
        curr_tmax = current.get('temp_max', None)
        curr_ws_kmh = (current['windspeed'] * 3.6) if current.get('windspeed', None) is not None else None
        similar_flags = []
        for _, r in df.iterrows():
            sim = False
            if pd.notna(r['temp_max']) and curr_tmax is not None:
                sim = abs(r['temp_max'] - curr_tmax) <= similar_tol_temp
            # require both wind values to be present to compare
            if pd.notna(r['windspeed_kmh']) and curr_ws_kmh is not None:
                sim = sim and (abs(r['windspeed_kmh'] - curr_ws_kmh) <= similar_tol_wind)
            similar_flags.append(sim)
        df['similar_to_current'] = similar_flags
        n_sim = int(df['similar_to_current'].sum())
        n_sim_success = int(df[(df['similar_to_current']) & (df['successful_hunt'])].shape[0])
        prob_sim = float(n_sim_success / n_sim) if n_sim > 0 else None
    else:
        df['similar_to_current'] = False
        n_sim = 0
        prob_sim = None

    return {
        'current': current,
        'history_df': df,
        'prob_good_given_weather': prob_good,
        'n_good': n_good,
        'prob_similar': prob_sim,
        'n_similar': n_sim
    }

@st.cache_data(ttl=60)
def load_data():
    """Load all necessary CSV files (with error handling and fallback computation)"""
    # Load coordinates (required)
    try:
        coordinates = pd.read_csv('Refuge_Coordinates.csv')
    except Exception as e:
        st.error(f"Could not load Refuge_Coordinates.csv: {e}")
        raise

    # Load aggregated master harvest data
    master_data = load_all_harvest_data()

    # Try to load precomputed historical baseline; if not available or empty, compute from master_data
    import os
    hb_path = 'Historical_Baseline.csv'
    if os.path.exists(hb_path) and os.path.getsize(hb_path) > 0:
        try:
            historical = pd.read_csv(hb_path)
            # If read but empty or lacks expected columns, fallback
            if historical.empty or 'Area Name' not in historical.columns:
                raise pd.errors.EmptyDataError
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            st.warning("Historical_Baseline.csv appears empty or malformed. Computing baseline from aggregated harvest data.")
            historical = compute_historical_baseline(master_data)
    else:
        st.warning("Historical_Baseline.csv missing or empty. Computing baseline from aggregated harvest data.")
        historical = compute_historical_baseline(master_data)

    return historical, coordinates, master_data

historical, coordinates, master_data = load_data()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def calculate_season_week(date):
    """Calculate season week from date (season starts Oct 1). Accepts date, datetime, or pandas Timestamp."""
    # normalize to pandas Timestamp to avoid mixed-type arithmetic
    date = pd.to_datetime(date)
    year = date.year
    if date.month >= 10:
        start_year = year
    else:
        start_year = year - 1
    season_start = pd.Timestamp(year=start_year, month=10, day=1)
    week = int(((date - season_start).days // 7) + 1)
    return week

def get_migration_timing(season_week):
    """
    Determine migration timing status based on season week.

    Returns: dict with status, description, and color
    """
    if season_week <= 5:
        return {
            'status': 'Early Season',
            'description': 'Local birds dominant, migration just beginning',
            'color': '#FFA726',  # Orange
            'emoji': '🍂'
        }
    elif season_week <= 12:
        return {
            'status': 'Peak Season',
            'description': 'Maximum migration activity, best hunting conditions',
            'color': '#4CAF50',  # Green
            'emoji': ''
        }
    else:
        return {
            'status': 'Late Season',
            'description': 'Migration winding down, hardy birds remaining',
            'color': '#42A5F5',  # Blue
            'emoji': '❄️'
        }

def calculate_success_rate(master_data_df, location, season_week):
    """
    Calculate success rate metrics (hunters per duck/goose).

    Returns: dict with hunters_per_duck, hunters_per_goose, total_hunters, total_ducks, total_geese
    """
    loc_week_df = master_data_df[
        (master_data_df['Area Name'].str.upper() == location.upper()) &
        (master_data_df['Season_Week'] == season_week)
    ]

    if loc_week_df.empty:
        return None

    # Handle different column name variations
    total_hunters = 0
    total_ducks = 0
    total_geese = 0

    for _, row in loc_week_df.iterrows():
        hunters = row.get('# of Hunters', row.get('Hunters', row.get('Hunter', 0)))
        ducks = row.get('# of Ducks', row.get('Ducks', row.get('Duck', 0)))
        geese = row.get('# of Geese', row.get('Geese', row.get('Goose', 0)))

        # Convert to numeric, handling any non-numeric values
        if pd.notna(hunters):
            try:
                total_hunters += float(hunters)
            except (ValueError, TypeError):
                pass
        if pd.notna(ducks):
            try:
                total_ducks += float(ducks)
            except (ValueError, TypeError):
                pass
        if pd.notna(geese):
            try:
                total_geese += float(geese)
            except (ValueError, TypeError):
                pass

    # Calculate ratios
    hunters_per_duck = total_hunters / total_ducks if total_ducks > 0 else None
    hunters_per_goose = total_hunters / total_geese if total_geese > 0 else None

    return {
        'hunters_per_duck': hunters_per_duck,
        'hunters_per_goose': hunters_per_goose,
        'total_hunters': total_hunters,
        'total_ducks': total_ducks,
        'total_geese': total_geese
    }

def create_historical_trend_chart(master_data_df, location, season_week):
    """
    Create an interactive historical trend chart showing harvest data over years.

    Returns: plotly figure object
    """
    # Filter data for the selected location and season week
    loc_data = master_data_df[
        (master_data_df['Area Name'].str.upper() == location.upper()) &
        (master_data_df['Season_Week'] == season_week)
    ].copy()

    if loc_data.empty:
        return None

    # Extract year from date column
    def extract_year(date_val):
        if pd.isna(date_val):
            return None
        try:
            if isinstance(date_val, str):
                parsed_date = pd.to_datetime(date_val, errors='coerce')
            else:
                parsed_date = pd.to_datetime(date_val)

            if pd.notna(parsed_date):
                # For dates in Oct-Dec, use that year; for Jan, use previous year as season year
                if parsed_date.month >= 10:
                    return parsed_date.year
                else:
                    return parsed_date.year - 1
        except:
            pass
        return None

    loc_data['Year'] = loc_data['Date'].apply(extract_year)
    loc_data = loc_data[pd.notna(loc_data['Year'])]

    if loc_data.empty:
        return None

    # Group by year and calculate averages
    yearly_data = loc_data.groupby('Year').agg({
        'Average Ducks': 'mean',
        'Average Geese': 'mean',
        '# of Hunters': 'mean'
    }).reset_index()

    # Convert to numeric
    yearly_data['Average Ducks'] = pd.to_numeric(yearly_data['Average Ducks'], errors='coerce')
    yearly_data['Average Geese'] = pd.to_numeric(yearly_data['Average Geese'], errors='coerce')
    yearly_data['# of Hunters'] = pd.to_numeric(yearly_data['# of Hunters'], errors='coerce')

    # Remove any rows with NaN values
    yearly_data = yearly_data.dropna(subset=['Average Ducks'])

    if yearly_data.empty:
        return None

    # Sort by year
    yearly_data = yearly_data.sort_values('Year')

    # Create the chart
    fig = go.Figure()

    # Add duck harvest line
    fig.add_trace(go.Scatter(
        x=yearly_data['Year'],
        y=yearly_data['Average Ducks'],
        mode='lines+markers',
        name='Ducks per Hunter',
        line=dict(color='#6C6E36', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Ducks: %{y:.2f}<extra></extra>'
    ))

    # Add goose harvest line if available
    if 'Average Geese' in yearly_data.columns and yearly_data['Average Geese'].notna().any():
        fig.add_trace(go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Average Geese'],
            mode='lines+markers',
            name='Geese per Hunter',
            line=dict(color='#6B7785', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>Geese: %{y:.2f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f'Historical Harvest Trends - Season Week {season_week}',
        xaxis_title='Season Year',
        yaxis_title='Average Harvest per Hunter',
        plot_bgcolor='#DFD3B5',
        paper_bgcolor='#DFD3B5',
        font=dict(family='Inter, sans-serif', color='#3B3B3B'),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=30, t=60, b=50),
        height=350
    )

    fig.update_xaxes(
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        dtick=1  # Show every year
    )

    fig.update_yaxes(
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True
    )

    return fig

def generate_7day_forecast(start_date, location, historical_df, master_data_df, coordinates_df):
    """
    Generate 7-day forecast predictions.

    Returns: list of prediction dictionaries for each day
    """
    forecast_days = []

    for day_offset in range(7):
        forecast_date = start_date + timedelta(days=day_offset)

        # Skip if outside hunting season (Oct 1 - Jan 31)
        if forecast_date.month < 2 or (forecast_date.month == 2 and forecast_date.day == 1):
            if forecast_date.month >= 10 or forecast_date.month == 1:
                prediction = predict_activity_with_weather(
                    forecast_date,
                    location,
                    historical_df,
                    master_data_df,
                    coordinates_df
                )

                if not prediction.get('error'):
                    forecast_days.append({
                        'date': forecast_date,
                        'day_name': forecast_date.strftime('%A'),
                        'date_str': forecast_date.strftime('%b %d'),
                        'prediction': prediction
                    })

    return forecast_days

def predict_activity(date, location, historical_df, master_data_df):
    """
    Predict hunting activity for a given date and location using master aggregated data.
    
    Returns:
        dict: Contains prediction data or error message
    """
    season_week = calculate_season_week(date)
    
    # Get historical baseline
    hist_row = historical_df[
        (historical_df['Area Name'].str.upper() == location.upper()) & 
        (historical_df['Season_Week'] == season_week)
    ]
    
    if hist_row.empty:
        return {
            'error': f"No historical data for {location} in Season Week {season_week}"
        }
    
    hist_avg = hist_row['15yr_Avg_Ducks'].iloc[0]
    top_species = hist_row['Top_Species'].iloc[0]

    # Get current year data from master aggregated data
    # Filter for the selected location and season week
    loc_week_df = master_data_df[
        (master_data_df['Area Name'].str.upper() == location.upper()) &
        (master_data_df['Season_Week'] == season_week)
    ]
    
    current_avg = loc_week_df['Average Ducks'].mean() if not loc_week_df.empty else 0.0
    
    # Determine activity level
    if current_avg > hist_avg:
        activity = 'High'
        trend = f"↑ Up {((current_avg - hist_avg) / hist_avg * 100):.1f}% from average"
    else:
        activity = 'Low'
        trend = f"↓ Down {((hist_avg - current_avg) / hist_avg * 100):.1f}% from average"
    
    return {
        'error': None,
        'activity': activity,
        'species': top_species,
        'historical_avg': hist_avg,
        'current_avg': current_avg,
        'trend': trend,
        'season_week': season_week
    }

# ============================================================================
# SIDEBAR - INPUTS
# ============================================================================

def display_no_historical_data(message):
    """Display a prominent, accessible notification when historical data is missing."""
    html = f"""
    <div role="alert" style="background:#fff3cd;color:#000;padding:12px;border-radius:8px;border:1px solid #ffeeba;font-weight:600">
    {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color: white;'>Forecast Settings</h2>", unsafe_allow_html=True)

# Calculate hunting season date range (Oct 1 - Jan 31)
today = datetime.today()
current_year = today.year

# Determine the hunting season based on current date
if today.month >= 10:  # Oct-Dec: current season
    season_start = datetime(current_year, 10, 1)
    season_end = datetime(current_year + 1, 1, 31)
elif today.month <= 1:  # Jan: ongoing season from last year
    season_start = datetime(current_year - 1, 10, 1)
    season_end = datetime(current_year, 1, 31)
else:  # Feb-Sep: next season
    season_start = datetime(current_year, 10, 1)
    season_end = datetime(current_year + 1, 1, 31)

date = st.sidebar.date_input(
    "Select Date",
    value=datetime.today() if season_start <= datetime.today() <= season_end else season_start,
    min_value=season_start,
    max_value=season_end,
    help="Choose the date you plan to hunt (Oct 1 - Jan 31)"
)

location = st.sidebar.selectbox(
    "Select Refuge/Area",
    sorted(coordinates['Area Name'].unique()),
    help="Choose your hunting location"
)

# Regulations Quick Reference
st.sidebar.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
with st.sidebar.expander("Quick Reference"):
    st.markdown("""
    <div style='color: white;'>

    <strong>2025-2026 Season Overview</strong><br><br>

    <strong>Duck Season:</strong><br>
    - October 25 - January 25, 2026<br>
    - Bag Limit: 7 ducks daily<br>
    - Mallard Limit: 7 (no more than 2 hens)<br>
    - Pintail Limit: 3<br>
    - Canvasback: 2<br>
    - Scaup: 2<br><br>

    <strong>Goose Season:</strong><br>
    - October 25 - January 25, 2026<br>
    - Dark Geese: 30 daily<br>
    - White Geese: No limit<br>
    - Brant: 2 daily<br><br>

    <strong>Shooting Hours:</strong><br>
    - 1/2 hour before sunrise to sunset<br>
    - Check specific refuge rules<br><br>

    <em>Always verify current regulations with CDFW before hunting.</em>

    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# WEATHER FUNCTIONS (must be defined before main content)
# ============================================================================

@st.cache_data(ttl=24*3600)
def fetch_comprehensive_weather(lat, lon, date):
    """
    Fetch comprehensive weather data including:
    - Temperature (max, min, average)
    - Wind (speed, direction)
    - Sunrise/Sunset times
    - Precipitation
    - Weather conditions

    For dates beyond forecast range, uses 5-year historical average.
    Returns dict with all weather metrics or None on error.
    """
    try:
        if pd.isna(lat) or pd.isna(lon):
            return None

        date = pd.to_datetime(date)
        date_str = date.strftime('%Y-%m-%d')
        today = pd.Timestamp.today().normalize()

        # Parameters for comprehensive weather data
        params = (
            "daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,windspeed_10m_max,winddirection_10m_dominant,"
            "sunrise,sunset,weathercode&"
            "temperature_unit=fahrenheit&windspeed_unit=mph&"
            "precipitation_unit=inch&timezone=America/Los_Angeles"
        )

        # Use forecast endpoint for today/future, archive for past
        if date >= today:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&{params}"
        else:
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&{params}"

        resp = requests.get(url, timeout=15)

        # If forecast is out of range (400 error), use historical average
        if resp.status_code == 400:
            return fetch_historical_weather_average(lat, lon, date)

        resp.raise_for_status()
        data = resp.json()

        daily = data.get('daily', {})
        times = daily.get('time', [])

        if not times:
            # If no data returned, try historical average
            return fetch_historical_weather_average(lat, lon, date)

        # Extract all weather data
        weather_data = {
            'date': pd.to_datetime(times[0]),
            'temp_max': daily.get('temperature_2m_max', [None])[0],
            'temp_min': daily.get('temperature_2m_min', [None])[0],
            'temp_mean': daily.get('temperature_2m_mean', [None])[0],
            'precipitation': daily.get('precipitation_sum', [None])[0],
            'windspeed': daily.get('windspeed_10m_max', [None])[0],
            'wind_direction': daily.get('winddirection_10m_dominant', [None])[0],
            'sunrise': daily.get('sunrise', [None])[0],
            'sunset': daily.get('sunset', [None])[0],
            'weathercode': daily.get('weathercode', [None])[0]
        }

        # Calculate average temp if mean not available
        if weather_data['temp_mean'] is None and weather_data['temp_max'] is not None and weather_data['temp_min'] is not None:
            weather_data['temp_mean'] = (weather_data['temp_max'] + weather_data['temp_min']) / 2

        # Convert wind direction to cardinal direction
        weather_data['wind_direction_cardinal'] = degrees_to_cardinal(weather_data['wind_direction'])

        # Parse sunrise/sunset times
        if weather_data['sunrise']:
            weather_data['sunrise_time'] = pd.to_datetime(weather_data['sunrise']).strftime('%I:%M %p')
        else:
            weather_data['sunrise_time'] = 'N/A'

        if weather_data['sunset']:
            weather_data['sunset_time'] = pd.to_datetime(weather_data['sunset']).strftime('%I:%M %p')
        else:
            weather_data['sunset_time'] = 'N/A'

        return weather_data

    except Exception as e:
        print(f"Error fetching weather: {e}")
        # Try historical average as fallback
        return fetch_historical_weather_average(lat, lon, date)

def fetch_historical_weather_average(lat, lon, date):
    """
    Fetch historical weather data for the same date over past 5 years
    and return the average. Used when forecast data is unavailable.
    """
    try:
        date = pd.to_datetime(date)
        historical_data = []

        # Fetch data from past 5 years for the same month/day
        for years_back in range(1, 6):
            try:
                # Calculate the past date (same month/day, previous year)
                past_date = date.replace(year=date.year - years_back)
            except ValueError:
                # Handle Feb 29 edge case
                past_date = date.replace(year=date.year - years_back, day=28)

            past_date_str = past_date.strftime('%Y-%m-%d')

            params = (
                "daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                "precipitation_sum,windspeed_10m_max,winddirection_10m_dominant,"
                "sunrise,sunset,weathercode&"
                "temperature_unit=fahrenheit&windspeed_unit=mph&"
                "precipitation_unit=inch&timezone=America/Los_Angeles"
            )

            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={past_date_str}&end_date={past_date_str}&{params}"

            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get('daily', {})
                if daily.get('time'):
                    historical_data.append(daily)

        if not historical_data:
            return None

        # Calculate averages from historical data
        temps_max = []
        temps_min = []
        temps_mean = []
        winds = []
        precips = []
        wind_dirs = []

        for daily in historical_data:
            if daily.get('temperature_2m_max') and daily.get('temperature_2m_max')[0] is not None:
                temps_max.append(daily['temperature_2m_max'][0])
            if daily.get('temperature_2m_min') and daily.get('temperature_2m_min')[0] is not None:
                temps_min.append(daily['temperature_2m_min'][0])
            if daily.get('temperature_2m_mean') and daily.get('temperature_2m_mean')[0] is not None:
                temps_mean.append(daily['temperature_2m_mean'][0])
            if daily.get('windspeed_10m_max') and daily.get('windspeed_10m_max')[0] is not None:
                winds.append(daily['windspeed_10m_max'][0])
            if daily.get('precipitation_sum') and daily.get('precipitation_sum')[0] is not None:
                precips.append(daily['precipitation_sum'][0])
            if daily.get('winddirection_10m_dominant') and daily.get('winddirection_10m_dominant')[0] is not None:
                wind_dirs.append(daily['winddirection_10m_dominant'][0])

        # Create weather data from averages
        avg_temp_max = round(np.mean(temps_max), 1) if temps_max else None
        avg_temp_min = round(np.mean(temps_min), 1) if temps_min else None
        avg_temp_mean = round(np.mean(temps_mean), 1) if temps_mean else None

        # If mean not available, calculate from max/min
        if avg_temp_mean is None and avg_temp_max is not None and avg_temp_min is not None:
            avg_temp_mean = round((avg_temp_max + avg_temp_min) / 2, 1)

        weather_data = {
            'date': date,
            'temp_max': avg_temp_max,
            'temp_min': avg_temp_min,
            'temp_mean': avg_temp_mean,
            'windspeed': round(np.mean(winds), 1) if winds else None,
            'wind_direction': round(np.mean(wind_dirs), 0) if wind_dirs else None,
            'precipitation': round(np.mean(precips), 2) if precips else None,
            'sunrise': None,  # Not critical for forecast
            'sunset': None,
            'sunrise_time': 'N/A',
            'sunset_time': 'N/A',
            'weathercode': None,
            'is_historical_average': True  # Flag to indicate this is historical avg
        }

        # Convert wind direction to cardinal
        weather_data['wind_direction_cardinal'] = degrees_to_cardinal(weather_data['wind_direction'])

        return weather_data

    except Exception as e:
        print(f"Error fetching historical average: {e}")
        return None

def degrees_to_cardinal(degrees):
    """Convert wind direction in degrees to cardinal direction (N, NE, E, etc.)"""
    if degrees is None or pd.isna(degrees):
        return 'Variable'

    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = int((degrees + 11.25) / 22.5) % 16
    return directions[index]

def calculate_weather_score(weather_data):
    """
    Calculate a weather favorability score (0-100) for duck hunting based on:
    - Temperature (ideal: 30-50°F)
    - Wind (ideal: 5-15 mph, favorable directions: N, NE, NW)
    - Precipitation (light is ok, heavy is bad)

    Returns: float score 0-100
    """
    if not weather_data:
        return 50.0  # Neutral score if no data

    score = 0
    factors = 0

    # Temperature scoring (0-35 points)
    temp_avg = weather_data.get('temp_mean')
    if temp_avg is not None:
        if 30 <= temp_avg <= 50:
            score += 35  # Ideal range
        elif 20 <= temp_avg < 30 or 50 < temp_avg <= 60:
            score += 25  # Good range
        elif 10 <= temp_avg < 20 or 60 < temp_avg <= 70:
            score += 15  # Acceptable
        else:
            score += 5   # Poor
        factors += 1

    # Wind scoring (0-35 points)
    windspeed = weather_data.get('windspeed')
    wind_dir = weather_data.get('wind_direction_cardinal', '')

    if windspeed is not None:
        # Wind speed scoring
        if 5 <= windspeed <= 15:
            wind_score = 25  # Ideal
        elif windspeed < 5:
            wind_score = 15  # Too calm
        elif 15 < windspeed <= 25:
            wind_score = 20  # Acceptable
        else:
            wind_score = 5   # Too windy

        # Wind direction bonus (favorable for migration)
        if any(d in wind_dir for d in ['N', 'NE', 'NW']):
            wind_score += 10

        score += wind_score
        factors += 1

    # Precipitation scoring (0-30 points)
    precip = weather_data.get('precipitation')
    if precip is not None:
        if precip == 0:
            score += 25  # Dry
        elif precip <= 0.1:
            score += 30  # Light drizzle can be good
        elif precip <= 0.3:
            score += 15  # Moderate
        else:
            score += 5   # Heavy rain
        factors += 1

    # Calculate final score
    if factors == 0:
        return 50.0

    # Normalize to 0-100 scale
    max_possible = 35 + 35 + 30  # temp + wind + precip
    final_score = (score / max_possible) * 100

    return round(final_score, 1)

def get_historical_weather_comparison(lat, lon, date, years_back=5):
    """
    Fetch weather for the same date over the past N years.
    Returns list of weather data dictionaries.
    """
    date = pd.to_datetime(date)
    historical_weather = []

    for year_offset in range(1, years_back + 1):
        try:
            # Go back N years
            past_date = date.replace(year=date.year - year_offset)
        except ValueError:
            # Handle Feb 29 edge case
            past_date = date.replace(year=date.year - year_offset, day=28)

        weather = fetch_comprehensive_weather(lat, lon, past_date)
        if weather:
            weather['year'] = past_date.year
            historical_weather.append(weather)

    return historical_weather

def calculate_historical_weather_averages(historical_weather):
    """Calculate averages from historical weather data."""
    if not historical_weather:
        return None

    temps_max = [w['temp_max'] for w in historical_weather if w.get('temp_max') is not None]
    temps_min = [w['temp_min'] for w in historical_weather if w.get('temp_min') is not None]
    temps_mean = [w['temp_mean'] for w in historical_weather if w.get('temp_mean') is not None]
    winds = [w['windspeed'] for w in historical_weather if w.get('windspeed') is not None]
    precips = [w['precipitation'] for w in historical_weather if w.get('precipitation') is not None]

    return {
        'avg_temp_max': round(np.mean(temps_max), 1) if temps_max else None,
        'avg_temp_min': round(np.mean(temps_min), 1) if temps_min else None,
        'avg_temp_mean': round(np.mean(temps_mean), 1) if temps_mean else None,
        'avg_windspeed': round(np.mean(winds), 1) if winds else None,
        'avg_precipitation': round(np.mean(precips), 2) if precips else None,
    }

def generate_hunting_recommendation(activity, weather_score, weather_data, current_avg, hist_avg):
    """Generate detailed hunting recommendation based on all factors."""

    recommendations = []

    # Base recommendation from activity
    if activity == 'High':
        recommendations.append("Excellent hunting conditions expected based on historical harvest data.")
    elif activity == 'Moderate':
        recommendations.append("Moderate hunting activity expected.")
    else:
        recommendations.append("Lower than average activity expected.")

    # Weather-based recommendations
    if weather_score >= 70:
        recommendations.append("Weather conditions are highly favorable for waterfowl hunting.")
    elif weather_score >= 50:
        recommendations.append("Weather conditions are acceptable for hunting.")
    else:
        recommendations.append("Weather conditions may be challenging.")

    # Specific weather factors
    if weather_data:
        temp = weather_data.get('temp_mean')
        wind = weather_data.get('windspeed')
        wind_dir = weather_data.get('wind_direction_cardinal', '')
        precip = weather_data.get('precipitation', 0)

        if temp and temp < 40:
            recommendations.append("Cold temperatures may increase duck activity.")

        if wind and 5 <= wind <= 15:
            recommendations.append("Wind conditions are ideal for decoy movement.")
        elif wind and wind > 20:
            recommendations.append("High winds may make hunting challenging.")

        if 'N' in wind_dir:
            recommendations.append("North winds are favorable for duck migration.")

        if precip and precip > 0.3:
            recommendations.append("Heavy precipitation expected - prepare accordingly.")

    return " ".join(recommendations)

def predict_activity_with_weather(date, location, historical_df, master_data_df, coordinates_df):
    """
    Enhanced prediction that incorporates weather data into hunting activity forecast.

    Returns:
        dict: Contains prediction data including weather analysis
    """
    season_week = calculate_season_week(date)

    # Get historical baseline
    hist_row = historical_df[
        (historical_df['Area Name'].str.upper() == location.upper()) &
        (historical_df['Season_Week'] == season_week)
    ]

    if hist_row.empty:
        return {
            'error': f"No historical data for {location} in Season Week {season_week}"
        }

    hist_avg = hist_row['15yr_Avg_Ducks'].iloc[0]
    top_species = hist_row['Top_Species'].iloc[0]

    # Get current year data from master aggregated data
    loc_week_df = master_data_df[
        (master_data_df['Area Name'].str.upper() == location.upper()) &
        (master_data_df['Season_Week'] == season_week)
    ]

    # Convert 'Average Ducks' to numeric, handling any string values
    if not loc_week_df.empty:
        current_avg = pd.to_numeric(loc_week_df['Average Ducks'], errors='coerce').mean()
        # If all values were non-numeric (resulting in NaN), default to 0.0
        if pd.isna(current_avg):
            current_avg = 0.0
    else:
        current_avg = 0.0

    # Get second most common species
    second_species = None
    if not loc_week_df.empty:
        # Try to get #2 species from the data
        if '#2 Species' in loc_week_df.columns:
            species_series = loc_week_df['#2 Species'].fillna(pd.NA)
            # Clean species codes (remove [CITE: xxx] patterns)
            species_series = species_series.astype(str).str.split('[').str[0].str.strip()
            # Filter out species that match top_species
            species_series = species_series[species_series != top_species]
            if not species_series.mode().empty:
                second_species = species_series.mode().iloc[0]
        elif '#2 Duck Species' in loc_week_df.columns:
            species_series = loc_week_df['#2 Duck Species'].fillna(pd.NA)
            # Clean species codes (remove [CITE: xxx] patterns)
            species_series = species_series.astype(str).str.split('[').str[0].str.strip()
            # Filter out species that match top_species
            species_series = species_series[species_series != top_species]
            if not species_series.mode().empty:
                second_species = species_series.mode().iloc[0]

    # Ensure second_species is actually different from top_species
    if second_species and str(second_species).upper() == str(top_species).upper():
        second_species = None

    # Get coordinates for weather data
    coords_row = coordinates_df[coordinates_df['Area Name'].str.upper() == location.upper()]

    weather_data = None
    weather_score = 50.0  # Default neutral score
    historical_weather = []
    historical_weather_avg = None

    if not coords_row.empty:
        lat = coords_row.iloc[0]['Latitude']
        lon = coords_row.iloc[0]['Longitude']

        # Fetch current/forecast weather
        weather_data = fetch_comprehensive_weather(lat, lon, date)

        # Calculate weather score
        weather_score = calculate_weather_score(weather_data)

        # Get historical weather for comparison (past 5 years)
        historical_weather = get_historical_weather_comparison(lat, lon, date, years_back=5)
        historical_weather_avg = calculate_historical_weather_averages(historical_weather)

    # Calculate base activity from harvest data
    if current_avg > hist_avg:
        harvest_score = 70  # Base score for high activity
    else:
        harvest_score = 50  # Base score for moderate/low activity

    # Combine harvest data and weather score (60% harvest, 40% weather)
    combined_score = (harvest_score * 0.6) + (weather_score * 0.4)

    # Determine final activity level based on combined score
    if combined_score >= 65:
        final_activity = 'High'
        activity_color = 'success'
        activity_emoji = ''
    elif combined_score >= 45:
        final_activity = 'Moderate'
        activity_color = 'info'
        activity_emoji = ''
    else:
        final_activity = 'Low'
        activity_color = 'warning'
        activity_emoji = ''

    # Generate trend description
    if current_avg > hist_avg:
        trend = f"↑ Up {((current_avg - hist_avg) / hist_avg * 100):.1f}% from average"
    elif current_avg < hist_avg:
        trend = f"↓ Down {((hist_avg - current_avg) / hist_avg * 100):.1f}% from average"
    else:
        trend = "→ On par with average"

    # Generate weather-influenced recommendation
    recommendation = generate_hunting_recommendation(
        final_activity,
        weather_score,
        weather_data,
        current_avg,
        hist_avg
    )

    return {
        'error': None,
        'activity': final_activity,
        'activity_color': activity_color,
        'activity_emoji': activity_emoji,
        'species': top_species,
        'second_species': second_species,
        'historical_avg': hist_avg,
        'current_avg': current_avg,
        'trend': trend,
        'season_week': season_week,
        'weather_data': weather_data,
        'weather_score': weather_score,
        'historical_weather': historical_weather,
        'historical_weather_avg': historical_weather_avg,
        'combined_score': combined_score,
        'harvest_score': harvest_score,
        'recommendation': recommendation
    }

def generate_print_report(date, location, prediction, master_data_df):
    """Generate a comprehensive HTML report for printing/download."""
    import base64
    from io import BytesIO

    # Try to load and encode the navbar image
    navbar_img_base64 = ""
    navbar_candidates = [
        'images/ff_navbar.png',
        'images/ff_navbar.jpg',
        'ff_navbar.png',
        'ff_navbar.jpg'
    ]

    for navbar_path in navbar_candidates:
        if os.path.exists(navbar_path):
            with open(navbar_path, 'rb') as f:
                navbar_img_base64 = base64.b64encode(f.read()).decode()
            break

    # Get historical data for the same season week across all years
    season_week = prediction['season_week']

    # Filter master_data for matching location and season week
    historical_records = master_data_df[
        (master_data_df['Area Name'].str.upper() == location.upper())
    ].copy()

    # Filter by matching season week if 'Season_Week' column exists
    if 'Season_Week' in historical_records.columns:
        historical_records = historical_records[
            historical_records['Season_Week'] == season_week
        ].copy()

        # Sort by date if available
        if 'Date' in historical_records.columns:
            # Convert Date column to datetime for sorting
            def parse_date_flexible(date_val):
                if pd.isna(date_val):
                    return pd.NaT
                if isinstance(date_val, pd.Timestamp):
                    return date_val

                date_str = str(date_val).strip()
                formats = ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y', '%m/%d/%y', '%m/%d']

                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                try:
                    return pd.to_datetime(date_str)
                except:
                    return pd.NaT

            historical_records['Date_parsed'] = historical_records['Date'].apply(parse_date_flexible)
            historical_records = historical_records.sort_values('Date_parsed', ascending=False)
            historical_records['Date'] = historical_records['Date_parsed']

    # Build the historical data table
    historical_table_rows = ""
    if not historical_records.empty:
        for _, row in historical_records.iterrows():
            # Handle date formatting
            if pd.notna(row.get('Date')):
                try:
                    if isinstance(row['Date'], str):
                        date_str = row['Date']
                    else:
                        date_str = row['Date'].strftime('%m/%d/%Y')
                except:
                    date_str = str(row.get('Date', 'N/A'))
            else:
                date_str = 'N/A'

            # Get all the data fields, handling different possible column names
            hunters = row.get('# of Hunters', row.get('Hunters', row.get('Hunter', 'N/A')))
            ducks = row.get('# of Ducks', row.get('Ducks', row.get('Duck', 'N/A')))
            geese = row.get('# of Geese', row.get('Geese', row.get('Goose', 'N/A')))
            avg_ducks = row.get('Average Ducks', row.get('Avg Ducks', row.get('Avg. Ducks', 'N/A')))
            avg_geese = row.get('Average Geese', row.get('Avg Geese', row.get('Avg. Geese', 'N/A')))
            species1 = get_species_full_name(row.get('#1 Species', row.get('Top_Species', 'N/A')))
            species2 = get_species_full_name(row.get('#2 Species', row.get('Second_Species', 'N/A')))

            # Format numeric values
            if pd.notna(hunters) and hunters != 'N/A':
                hunters = f"{int(float(hunters))}" if str(hunters).replace('.','').isdigit() else hunters
            if pd.notna(ducks) and ducks != 'N/A':
                ducks = f"{int(float(ducks))}" if str(ducks).replace('.','').isdigit() else ducks
            if pd.notna(geese) and geese != 'N/A':
                geese = f"{int(float(geese))}" if str(geese).replace('.','').isdigit() else geese
            if pd.notna(avg_ducks) and avg_ducks != 'N/A':
                avg_ducks = f"{float(avg_ducks):.2f}" if str(avg_ducks).replace('.','').replace('-','').isdigit() else avg_ducks
            if pd.notna(avg_geese) and avg_geese != 'N/A':
                avg_geese = f"{float(avg_geese):.2f}" if str(avg_geese).replace('.','').replace('-','').isdigit() else avg_geese

            historical_table_rows += f"""
            <tr>
                <td>{date_str}</td>
                <td>{hunters}</td>
                <td>{ducks}</td>
                <td>{geese}</td>
                <td>{avg_ducks}</td>
                <td>{avg_geese}</td>
                <td>{species1}</td>
                <td>{species2}</td>
            </tr>
            """
    else:
        historical_table_rows = "<tr><td colspan='8' style='text-align:center;'>No historical data available for this date</td></tr>"

    # Get weather data
    weather_data = prediction.get('weather_data', {})
    weather_score = prediction.get('weather_score', 0)

    # Build HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Flyway Forecast Report - {location} - {date.strftime('%B %d, %Y')}</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Montserrat:wght@400;600;700&display=swap');
            body {{
                font-family: 'Inter', Arial, sans-serif;
                max-width: 1000px;
                margin: 20px auto;
                padding: 20px;
                color: #3B3B3B;
                background: #DFD3B5;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .header img {{
                max-width: 400px;
                margin-bottom: 20px;
            }}
            h1 {{
                font-family: 'Montserrat', Arial, sans-serif;
                color: #6C6E36;
                font-size: 32px;
                margin: 10px 0;
            }}
            h2 {{
                font-family: 'Montserrat', Arial, sans-serif;
                color: #6C6E36;
                font-size: 24px;
                margin: 20px 0 10px 0;
                border-bottom: 2px solid #6C6E36;
                padding-bottom: 8px;
            }}
            .forecast-summary {{
                background: linear-gradient(135deg, #6C6E36 0%, #7a7d3e 100%);
                color: white;
                padding: 24px;
                border-radius: 12px;
                margin-bottom: 24px;
            }}
            .forecast-summary h3 {{
                margin: 0 0 12px 0;
                font-size: 28px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-box {{
                background: white;
                padding: 16px;
                border-radius: 8px;
                border-left: 4px solid #6C6E36;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 8px;
            }}
            .metric-value {{
                font-size: 32px;
                font-weight: 700;
                color: #6C6E36;
            }}
            .metric-subtitle {{
                font-size: 12px;
                color: #666;
                margin-top: 4px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background: #6C6E36;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                font-size: 13px;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
                font-size: 14px;
            }}
            tr:nth-child(even) {{
                background: #f5f5f5;
            }}
            .info-box {{
                background: #e7f3ff;
                border-left: 4px solid #2196F3;
                padding: 12px 16px;
                border-radius: 6px;
                margin: 16px 0;
                font-size: 13px;
                color: #1565C0;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
            @media print {{
                body {{
                    background: white;
                }}
                .metric-box {{
                    break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            {'<img src="data:image/png;base64,' + navbar_img_base64 + '" alt="Flyway Forecast">' if navbar_img_base64 else '<h1>FLYWAY FORECAST</h1>'}
        </div>

        <div class="forecast-summary">
            <h3>{location}</h3>
            <p style="margin: 8px 0 0 0; font-size: 18px; opacity: 0.95;">
                {date.strftime('%B %d, %Y')} | Season Week {prediction['season_week']} | {prediction['activity']} Activity Expected
            </p>
            <p style="margin: 12px 0 0 0; font-size: 16px;">
                Overall Score: <strong style="font-size: 24px;">{prediction['combined_score']:.0f}/100</strong>
            </p>
        </div>

        <h2>Forecast Summary</h2>
        <div style="background: #f5f5f5; padding: 16px; border-left: 4px solid #6C6E36; border-radius: 8px; margin: 16px 0;">
            <strong>Hunter's Forecast:</strong><br>
            {prediction.get('recommendation', 'Check conditions and plan accordingly.')}
        </div>

        <h2>Harvest Data Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Historical Average</div>
                <div class="metric-value">{prediction['historical_avg']:.2f}</div>
                <div class="metric-subtitle">15-year average</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Current Average</div>
                <div class="metric-value">{prediction['current_avg']:.2f}</div>
                <div class="metric-subtitle">{prediction['trend']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Harvest Score</div>
                <div class="metric-value">{prediction['harvest_score']:.0f}</div>
                <div class="metric-subtitle">out of 100</div>
            </div>
        </div>

        <div style="margin: 16px 0;">
            <strong>Primary Species:</strong> {get_species_full_name(prediction['species'])}<br>
            {'<strong>Secondary Species:</strong> ' + get_species_full_name(prediction.get('second_species')) if prediction.get('second_species') else ''}
        </div>

        <h2>Weather Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Weather Score</div>
                <div class="metric-value">{weather_score:.0f}</div>
                <div class="metric-subtitle">out of 100</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Temperature</div>
                <div class="metric-value">{weather_data.get('temp_mean', 0):.0f}°F</div>
                <div class="metric-subtitle">High: {weather_data.get('temp_max', 0):.0f}°F | Low: {weather_data.get('temp_min', 0):.0f}°F</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Wind</div>
                <div class="metric-value">{weather_data.get('windspeed', 0):.0f}</div>
                <div class="metric-subtitle">mph {weather_data.get('wind_direction_cardinal', 'N/A')}</div>
            </div>
        </div>

        {'<div class="info-box">ℹ️ Weather data based on 5-year historical average for this date</div>' if weather_data and weather_data.get('is_historical_average') else ''}

        <h2>Historical Data - Season Week {prediction['season_week']} at {location}</h2>
        <p style="color: #666; font-size: 14px; margin-bottom: 12px;">
            Showing all recorded data for Season Week {prediction['season_week']} across all years at this location. This matches the same point in the hunting season rather than the same calendar date.
        </p>

        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Hunters</th>
                    <th>Ducks</th>
                    <th>Geese</th>
                    <th>Avg Ducks</th>
                    <th>Avg Geese</th>
                    <th>#1 Species</th>
                    <th>#2 Species</th>
                </tr>
            </thead>
            <tbody>
                {historical_table_rows}
            </tbody>
        </table>

        <div class="footer">
            <p><strong>Flyway Forecast</strong> | flywayforecast1@gmail.com</p>
            <p>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p style="font-size: 11px; margin-top: 8px;">
                Data Source: 15+ years of California waterfowl harvest data | Weather data from Open-Meteo API
            </p>
        </div>
    </body>
    </html>
    """

    return html_content

def display_weather_card(weather_data, weather_score, top_species=None, second_species=None, success_rate=None):
    """Display weather and species information in a simple, clear format."""

    if not weather_data:
        # Show informative message when weather data is unavailable
        st.markdown(f"""
        <div style="
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 16px 24px;
            border-radius: 8px;
            margin-bottom: 16px;
        ">
            <div style="color: #856404; font-weight: 600; font-size: 14px; margin-bottom: 6px;">
                ⚠️ Weather Data Unavailable
            </div>
            <div style="color: #856404; font-size: 14px; line-height: 1.5;">
                Forecast is based on 15-year historical harvest data only. Weather conditions could not be retrieved for this location and date.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Still display species information if available
        if top_species or second_species:
            st.markdown("<div style='margin-top: 12px; color: black; margin-bottom: 8px;'><strong style='color: black;'>Expected Species:</strong></div>", unsafe_allow_html=True)

            if top_species:
                # Show #1 species
                species1_full = get_species_full_name(top_species)
                st.markdown(f"""
                <div style='color: black; font-size: 15px; margin-bottom: 6px;'>
                    <span style='font-weight: 600; color: #6C6E36;'>#1:</span> {species1_full}
                </div>
                """, unsafe_allow_html=True)

                # Show #2 species or N/A
                if second_species and str(second_species).strip():
                    species2_full = get_species_full_name(second_species)
                else:
                    species2_full = "N/A"

                st.markdown(f"""
                <div style='color: black; font-size: 15px;'>
                    <span style='font-weight: 600; color: #6C6E36;'>#2:</span> {species2_full}
                </div>
                """, unsafe_allow_html=True)
        return

    # Weather score color coding
    if weather_score >= 70:
        score_label = "Excellent"
    elif weather_score >= 50:
        score_label = "Good"
    else:
        score_label = "Fair"

    # Display metrics in columns with explicit black text
    w1, w2, w3 = st.columns(3)

    with w1:
        st.markdown(f"""
        <div style="color: black;">
            <div style="font-size: 14px; color: black; font-weight: 500;">Weather Score</div>
            <div style="font-size: 36px; font-weight: 600; color: black;">{weather_score:.0f}/100</div>
            <div style="font-size: 14px; color: #28a745; font-weight: 600;">{score_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with w2:
        st.markdown(f"""
        <div style="color: black;">
            <div style="font-size: 14px; color: black; font-weight: 500;">Temperature</div>
            <div style="font-size: 36px; font-weight: 600; color: black;">{weather_data.get('temp_mean', 0):.0f}°F</div>
            <div style="font-size: 12px; color: #666;">High: {weather_data.get('temp_max', 0):.0f}°F | Low: {weather_data.get('temp_min', 0):.0f}°F</div>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        wind_dir = weather_data.get('wind_direction_cardinal', 'N/A')
        wind_speed = weather_data.get('windspeed', 0)
        st.markdown(f"""
        <div style="color: black;">
            <div style="font-size: 14px; color: black; font-weight: 500;">Wind</div>
            <div style="font-size: 36px; font-weight: 600; color: black;">{wind_speed:.0f}</div>
            <div style="font-size: 14px; color: #666;">mph {wind_dir}</div>
        </div>
        """, unsafe_allow_html=True)

    # Show note if using historical average
    # Display top species if available
    if top_species or second_species:
        # Build success rate text if available
        success_text = ""
        if success_rate and success_rate.get('hunters_per_duck'):
            hpd = success_rate['hunters_per_duck']
            success_text = f" | <span style='font-size: 13px; color: #666;'>Success: {hpd:.1f} hunters/duck</span>"

        st.markdown(f"<div style='margin-top: 20px; color: black; margin-bottom: 8px;'><strong style='color: black;'>Expected Species:</strong>{success_text}</div>", unsafe_allow_html=True)

        if top_species:
            # Show #1 species
            species1_full = get_species_full_name(top_species)
            st.markdown(f"""
            <div style='color: black; font-size: 15px; margin-bottom: 6px;'>
                <span style='font-weight: 600; color: #6C6E36;'>#1:</span> {species1_full}
            </div>
            """, unsafe_allow_html=True)

            # Show #2 species or N/A
            if second_species and str(second_species).strip():
                species2_full = get_species_full_name(second_species)
            else:
                species2_full = "N/A"

            st.markdown(f"""
            <div style='color: black; font-size: 15px;'>
                <span style='font-weight: 600; color: #6C6E36;'>#2:</span> {species2_full}
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT - COMPREHENSIVE LANDSCAPE LAYOUT
# ============================================================================

st.markdown('<a id="forecast"></a>', unsafe_allow_html=True)

# Compute prediction once
prediction = predict_activity_with_weather(date, location, historical, master_data, coordinates)

if prediction.get('error'):
    err = prediction['error']
    if err and 'no historical data' in err.lower():
        display_no_historical_data(err)
    else:
        st.warning(err)
else:
    # HERO SECTION - Full width forecast banner
    activity = prediction['activity']
    activity_emoji = prediction['activity_emoji']
    species = prediction['species']
    combined_score = prediction['combined_score']

    # Determine score box color based on score (greener = better, greyer = worse)
    if combined_score >= 70:
        score_box_color = "rgba(34, 139, 34, 0.85)"  # Forest green
    elif combined_score >= 55:
        score_box_color = "rgba(76, 175, 80, 0.75)"  # Medium green
    elif combined_score >= 40:
        score_box_color = "rgba(255, 193, 7, 0.65)"  # Amber/yellow
    else:
        score_box_color = "rgba(158, 158, 158, 0.65)"  # Grey

    hero_html = f"""
    <div style="
        background: linear-gradient(135deg, #6C6E36 0%, #7a7d3e 100%);
        border-radius: 12px;
        padding: 24px 48px;
        color: white;
        margin: 16px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 40px;">
            <div style="flex: 1;">
                <h1 style="margin: 0; color: white; font-size: 42px;">
                    {activity_emoji} {location}
                </h1>
                <h2 style="margin: 6px 0 0 0; color: white; font-size: 26px; font-weight: 400;">
                    {activity} Activity Expected
                </h2>
                <p style="margin: 8px 0 0 0; font-size: 17px; opacity: 0.95;">
                    Primarily <strong>{species}</strong> | Season Week {prediction['season_week']}
                </p>
            </div>
            <div style="text-align: center; padding: 16px 32px; background: {score_box_color}; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 6px;">Overall Score</div>
                <div style="font-size: 48px; font-weight: bold; line-height: 1;">{combined_score:.0f}</div>
                <div style="font-size: 13px; opacity: 0.8;">/100</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # RECOMMENDATION BOX - More compact
    st.markdown(f"""
    <div style="
        background: #f5f5f5;
        border-left: 4px solid #6C6E36;
        padding: 16px 32px;
        border-radius: 8px;
        margin: 12px 0 16px 0;
    ">
        <div style="color: #6C6E36; font-weight: 700; font-size: 14px; margin-bottom: 8px;">
            HUNTER'S FORECAST
        </div>
        <div style="color: #3B3B3B; font-size: 16px; line-height: 1.6;">
            {prediction.get('recommendation', 'Check conditions and good luck out there.')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Calculate success rate for later display
    success_rate = calculate_success_rate(master_data, location, prediction['season_week'])

    # TWO-COLUMN LAYOUT: Metrics & Weather Side-by-Side
    data_col1, data_col2 = st.columns([1, 1])

    with data_col1:
        st.markdown("<h3 style='color: black;'>Harvest Data Analysis</h3>", unsafe_allow_html=True)

        # Three metric columns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style="color: black;">
                <div style="font-size: 14px; color: black; font-weight: 500;">Historical Avg</div>
                <div style="font-size: 36px; font-weight: 600; color: black;">{prediction['historical_avg']:.2f}</div>
                <div style="font-size: 10px; color: #888; font-style: italic; margin-top: 4px;">
                    Avg ducks/hunter past 15 yrs
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            curr_delta = prediction['current_avg'] - prediction['historical_avg']
            delta_color = "#28a745" if curr_delta > 0 else "#dc3545"
            delta_symbol = "▲" if curr_delta > 0 else "▼"
            st.markdown(f"""
            <div style="color: black;">
                <div style="font-size: 14px; color: black; font-weight: 500;">Current Avg</div>
                <div style="font-size: 36px; font-weight: 600; color: black;">{prediction['current_avg']:.2f}</div>
                <div style="font-size: 10px; color: #888; font-style: italic; margin-top: 4px;">
                    Avg ducks/hunter this season
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div style="color: black;">
                <div style="font-size: 14px; color: black; font-weight: 500;">Harvest Score</div>
                <div style="font-size: 36px; font-weight: 600; color: black;">{prediction['harvest_score']:.0f}</div>
                <div style="font-size: 10px; color: #888; font-style: italic; margin-top: 4px;">
                    70=above avg, 50=below avg
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"<div style='margin-top: 15px; color: black;'><strong style='color: black;'>Trend:</strong> {prediction['trend']}</div>", unsafe_allow_html=True)

        # Generate print report HTML
        report_html = generate_print_report(date, location, prediction, master_data)
        report_filename = f"Flyway_Forecast_Print_Report_{location.replace(' ', '_')}_{date.strftime('%Y-%m-%d')}.html"

        # Add Print Report button - prominent styling
        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #6C6E36 0%, #7a7d3e 100%) !important;
            color: white !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            padding: 16px 32px !important;
            border-radius: 8px !important;
            border: 2px solid #6C6E36 !important;
            box-shadow: 0 4px 12px rgba(108, 110, 54, 0.3) !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stDownloadButton"] > button *,
        div[data-testid="stDownloadButton"] > button p,
        div[data-testid="stDownloadButton"] > button span,
        div[data-testid="stDownloadButton"] > button div,
        div[data-testid="stDownloadButton"] button[kind="secondary"] {
            color: white !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(108, 110, 54, 0.4) !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            st.download_button(
                label="📄 Print Forecast Report",
                data=report_html,
                file_name=report_filename,
                mime="text/html",
                help="Download a comprehensive report with all forecast data and historical records"
            )

    with data_col2:
        st.markdown("<h3 style='color: black;'>Weather Analysis</h3>", unsafe_allow_html=True)
        display_weather_card(
            prediction['weather_data'],
            prediction['weather_score'],
            prediction['species'],
            prediction.get('second_species', None),
            success_rate
        )

    # ============================================================================
    # MULTI-AREA COMPARISON TOOL
    # ============================================================================

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

    # Large, prominent heading for comparison
    st.markdown("<h2 style='color: black; font-size: 28px; margin: 0;'>Compare Multiple Areas</h2>", unsafe_allow_html=True)

    # Custom CSS for expander color
    st.markdown("""
    <style>
    div[data-testid="stExpander"] {
        background-color: #f5f5f5;
        border-left: 4px solid #6C6E36;
        border-radius: 8px;
        padding: 4px;
    }
    div[data-testid="stExpander"] summary {
        background-color: #f5f5f5;
        color: #6C6E36;
        font-weight: 600;
        border-radius: 6px;
    }
    div[data-testid="stExpander"] > div {
        background-color: white;
        border-radius: 6px;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("Select Areas", expanded=False):

        # Get all available areas
        all_areas = sorted(coordinates['Area Name'].unique())

        # Black text label
        st.markdown("<div style='color: black; font-weight: 500; margin-bottom: 8px;'>Select area to compare</div>", unsafe_allow_html=True)

        # Multiselect for area comparison (limit to 2 for simplicity)
        comparison_areas = st.multiselect(
            "Select area to compare",
            options=[area for area in all_areas if area != location],
            default=[],
            max_selections=2,
            help="Choose up to 2 areas to compare",
            label_visibility="collapsed"
        )

        if comparison_areas:
            # Add current location to comparison
            areas_to_compare = [location] + comparison_areas

            # Create comparison data
            comparison_data = []

            for area in areas_to_compare:
                area_prediction = predict_activity_with_weather(date, area, historical, master_data, coordinates)

                if not area_prediction.get('error'):
                    comparison_data.append({
                        'area': area,
                        'prediction': area_prediction
                    })

            if comparison_data:
                # Display comparison in clean columns
                comp_cols = st.columns(len(comparison_data))

                for idx, area_data in enumerate(comparison_data):
                    area_name = area_data['area']
                    pred = area_data['prediction']
                    is_primary = (area_name == location)

                    weather = pred.get('weather_data')
                    temp = weather.get('temp_mean', 0) if weather else 0
                    wind = weather.get('windspeed', 0) if weather else 0
                    wind_dir = weather.get('wind_direction_cardinal', 'N/A') if weather else 'N/A'

                    with comp_cols[idx]:
                        # Show area name
                        st.markdown(f"### {area_name}")

                        st.markdown("---")

                        # Compact display with proper indentation
                        st.markdown(f"""
**Harvest Data:**
&nbsp;&nbsp;&nbsp;&nbsp;Current Avg: {pred['current_avg']:.2f} ducks/hunter
&nbsp;&nbsp;&nbsp;&nbsp;Historical Avg: {pred['historical_avg']:.2f} ducks/hunter
&nbsp;&nbsp;&nbsp;&nbsp;Harvest Score: {pred['harvest_score']:.0f}/100

**Weather:**
&nbsp;&nbsp;&nbsp;&nbsp;Temperature: {temp:.0f}°F
&nbsp;&nbsp;&nbsp;&nbsp;Wind: {wind:.0f} mph {wind_dir}
&nbsp;&nbsp;&nbsp;&nbsp;Weather Score: {pred['weather_score']:.0f}/100

**Overall: {pred['combined_score']:.0f}/100**
                        """, unsafe_allow_html=True)

                # Find and show best area
                best_area = max(comparison_data, key=lambda x: x['prediction']['combined_score'])
                best_name = best_area['area']
                best_score = best_area['prediction']['combined_score']

                st.markdown(f"<div style='margin-top: 12px; color: black; font-size: 15px;'><strong>Best Option: {best_name}</strong> with score of {best_score:.0f}/100</div>", unsafe_allow_html=True)

            else:
                st.warning("Unable to generate comparison data.")

# ============================================================================
# MAP VIEW - Full Width
# ============================================================================

st.markdown('<a id="map"></a>', unsafe_allow_html=True)
st.subheader("All California Refuges (Season Week {})".format(
    calculate_season_week(date)
))

# Filter baseline for the week
week_num = calculate_season_week(date)
week_data = historical[historical['Season_Week'] == week_num]

# Get coordinates
coords_dict = dict(zip(coordinates['Area Name'], zip(coordinates['Latitude'], coordinates['Longitude'])))

# Debug: Check if Los Banos is in coords_dict
if 'LOS BANOS' in coords_dict:
    print(f"DEBUG: Los Banos found at coordinates: {coords_dict['LOS BANOS']}")
else:
    print(f"DEBUG: Los Banos NOT found in coords_dict. Available areas: {list(coords_dict.keys())[:10]}")

# Create map centered on California
m = folium.Map(
    location=[37.5, -119.5],
    zoom_start=6,
    tiles="OpenStreetMap"
)

# Add markers for all refuges
for area, (lat, lon) in coords_dict.items():
    # Check if this area has data for the selected week
    area_data = week_data[week_data['Area Name'] == area]

    if not area_data.empty:
        avg_ducks = area_data.iloc[0]['15yr_Avg_Ducks']

        # Color based on avg ducks
        if avg_ducks >= 5:
            color = 'darkgreen'
        elif avg_ducks >= 2:
            color = 'green'
        else:
            color = 'orange'

        popup_text = f"""
        <b>{area}</b><br>
        Avg Ducks: {avg_ducks:.2f}
        """
    else:
        # No data available for this week
        color = 'gray'
        popup_text = f"""
        <b>{area}</b><br>
        No data available
        """

    # Highlight selected location
    if area.upper() == location.upper():
        color = 'blue'

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=260),
        tooltip=area,
        icon=folium.Icon(color=color, prefix='fa', icon='duck')
    ).add_to(m)

# Display map - full width for better landscape view
st_folium(m, width=None, height=550)

# Map legend immediately below with no spacing
legend_html = """
<div style="background: white; padding: 12px 24px; border-radius: 8px; border: 1px solid #ddd; margin-top: -8px; margin-bottom: 16px;">
    <div style="font-weight: 600; margin-bottom: 8px; color: #6C6E36; font-size: 14px;">Map Legend</div>
    <div style="display: flex; gap: 24px; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 12px; height: 12px; background: green; border-radius: 50%;"></div>
            <span style="font-size: 14px;">High Activity (65-100)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 12px; height: 12px; background: orange; border-radius: 50%;"></div>
            <span style="font-size: 14px;">Moderate Activity (45-64)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 12px; height: 12px; background: red; border-radius: 50%;"></div>
            <span style="font-size: 14px;">Low Activity (&lt;45)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 12px; height: 12px; background: blue; border-radius: 50%;"></div>
            <span style="font-size: 14px;">Selected Location</span>
        </div>
    </div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# ============================================================================
# BOTTOM SECTIONS - ABOUT, MISSION & CONTACT
# ============================================================================

# Two-column layout for About and Mission
about_col1, about_col2 = st.columns(2)

with about_col1:
    st.markdown('<a id="about"></a>', unsafe_allow_html=True)
    st.markdown("### About Flyway Forecast")
    st.markdown("""
    Flyway Forecast combines over 15 years of comprehensive California waterfowl harvest data with advanced weather analytics to deliver the most accurate hunting forecasts available. Our platform analyzes historical hunting reports from state-operated areas—including hunter counts, waterfowl harvest totals, and species-specific data—alongside real-time and forecasted weather conditions.

    By integrating 5-year historical weather patterns with current forecasts, we calculate temperature, wind speed, wind direction, and precipitation trends that directly impact waterfowl behavior. Our proprietary scoring system weighs both harvest data (60%) and weather conditions (40%) to generate an Overall Score for each hunt, identifying which duck and goose species are most likely to be abundant and when conditions are optimal.
    """)

with about_col2:
    st.markdown('<a id="our-goal"></a>', unsafe_allow_html=True)
    st.markdown("### Our Mission")
    st.markdown("""
    To empower every California waterfowl hunter with reliable, data-driven insights, helping them make informed decisions, improve their success rates, and enjoy a safe and rewarding hunting experience across the state.

    With a commitment to accuracy and clarity, Flyway Forecast transforms historical data into predictive insights, empowering hunters with a data-driven edge for every waterfowl season.
    """)

# ============================================================================
# FOOTER - CONTACT & DATA INFO
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

footer_html = """
<div style="text-align: center; padding: 24px 0; color: #6B7785;">
    <div style="font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #6C6E36;">
        Contact Us
    </div>
    <div style="font-size: 16px; margin-bottom: 20px;">
        <a href="mailto:flywayforecast1@gmail.com" style="color: #6C6E36; text-decoration: none; font-weight: 500;">
            flywayforecast1@gmail.com
        </a>
    </div>
    <div style="font-size: 14px; opacity: 0.8; margin-bottom: 6px;">
        Data Source: 15 years of California waterfowl harvest data (2006-2024)
    </div>
    <div style="font-size: 14px; opacity: 0.8;">
        Season runs October 1 - December 31 | Weeks calculated from season start
    </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

# Bottom Image - Centered at 1/3 width
bottom_image_candidates = [
    'images/ff_image.png',
    'images/ff_image.jpg',
    'ff_image.png',
    'ff_image.jpg'
]

# Display bottom image centered at 1/10 width
bottom_image_found = False
for image_path in bottom_image_candidates:
    if os.path.exists(image_path):
        # Create columns to center the image at 1/10 width
        left_spacer, bottom_img_col, right_spacer = st.columns([4.5, 1, 4.5])
        with bottom_img_col:
            st.image(image_path, use_column_width=True)
        bottom_image_found = True
        break

if not bottom_image_found:
    st.warning("Bottom image 'ff_image.png' not found. Please add it to the images/ folder.")
