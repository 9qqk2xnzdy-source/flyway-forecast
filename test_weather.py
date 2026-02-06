import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.title("🌤️ Weather API Test")
st.write("This will test if the weather integration is working")

@st.cache_data(ttl=24*3600)
def fetch_comprehensive_weather(lat, lon, date):
    """Test weather fetch"""
    try:
        date = pd.to_datetime(date)
        date_str = date.strftime('%Y-%m-%d')
        today = pd.Timestamp.today().normalize()
        
        params = (
            "daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,windspeed_10m_max,winddirection_10m_dominant,"
            "sunrise,sunset&"
            "temperature_unit=fahrenheit&windspeed_unit=mph&"
            "precipitation_unit=inch&timezone=America/Los_Angeles"
        )
        
        if date >= today:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&{params}"
        else:
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&{params}"
        
        st.info(f"Fetching from: {url[:100]}...")
        
        resp = requests.get(url, timeout=15)
        data = resp.json()
        
        daily = data.get('daily', {})
        
        weather = {
            'temp_max': daily.get('temperature_2m_max', [None])[0],
            'temp_min': daily.get('temperature_2m_min', [None])[0],
            'temp_mean': daily.get('temperature_2m_mean', [None])[0],
            'windspeed': daily.get('windspeed_10m_max', [None])[0],
            'precipitation': daily.get('precipitation_sum', [None])[0],
        }
        
        if weather['temp_mean'] is None and weather['temp_max'] and weather['temp_min']:
            weather['temp_mean'] = (weather['temp_max'] + weather['temp_min']) / 2
        
        return weather
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# Input
date = st.date_input("Select Date", value=datetime.today())

location = st.selectbox("Select Location", [
    "Colusa Wildlife Area",
    "Los Banos Wildlife Area", 
    "Woodland Area"
])

# Coordinates
coords = {
    "Colusa Wildlife Area": (39.2141, -122.0094),
    "Los Banos Wildlife Area": (37.0580, -120.8499),
    "Woodland Area": (38.6785, -121.7733)
}

lat, lon = coords[location]

st.write(f"Testing for: **{location}**")
st.write(f"Coordinates: {lat}, {lon}")
st.write(f"Date: {date}")

if st.button("🔍 Fetch Weather", type="primary"):
    with st.spinner("Fetching weather data..."):
        weather = fetch_comprehensive_weather(lat, lon, date)
        
        if weather:
            st.success("✅ Weather data fetched successfully!")
            
            # Display weather
            st.markdown("### Weather Results:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Temperature",
                    f"{weather.get('temp_mean', 0):.1f}°F",
                    f"H: {weather.get('temp_max', 0):.0f}°F L: {weather.get('temp_min', 0):.0f}°F"
                )
            
            with col2:
                st.metric(
                    "Wind Speed",
                    f"{weather.get('windspeed', 0):.1f} mph"
                )
            
            with col3:
                st.metric(
                    "Precipitation",
                    f"{weather.get('precipitation', 0):.2f}\""
                )
            
            st.json(weather)
            
            st.success("✅ If you see data above, the weather API is working!")
            st.info("The problem is likely in how you integrated it into app.py")
            
        else:
            st.error("❌ Failed to fetch weather")
            st.warning("Check your internet connection or API availability")

st.markdown("---")
st.markdown("### Next Steps:")
st.markdown("""
If this test works but your main app doesn't show weather:

1. **Check your app.py** - Make sure you:
   - Added the imports at the top
   - Replaced the `predict_activity` function
   - Updated the display section

2. **Restart Streamlit** - After editing app.py:
   - Press Ctrl+C to stop
   - Run `streamlit run app.py` again

3. **Check for errors** - Look at the terminal for error messages

4. **Still not working?** Send me:
   - Any error messages you see
   - The first 20 lines of your app.py
   - Screenshot of what you see on the website
""")
                return pd.NaT