import requests
import pandas as pd
import numpy as np
from datetime import datetime

def degrees_to_cardinal(degrees):
    """Convert wind direction in degrees to cardinal direction (N, NE, E, etc.)"""
    if degrees is None or pd.isna(degrees):
        return 'Variable'

    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = int((degrees + 11.25) / 22.5) % 16
    return directions[index]

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

            print(f"Fetching {past_date.year}: {url[:100]}...")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get('daily', {})
                if daily.get('time'):
                    historical_data.append(daily)
                    print(f"  ✓ Got data for {past_date.year}")
            else:
                print(f"  ✗ Failed for {past_date.year}: {resp.status_code}")

        if not historical_data:
            print("\n✗ No historical data found")
            return None

        print(f"\n✓ Successfully fetched {len(historical_data)} years of historical data")

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
            'sunrise': None,
            'sunset': None,
            'sunrise_time': 'N/A',
            'sunset_time': 'N/A',
            'weathercode': None,
            'is_historical_average': True
        }

        # Convert wind direction to cardinal
        weather_data['wind_direction_cardinal'] = degrees_to_cardinal(weather_data['wind_direction'])

        return weather_data

    except Exception as e:
        print(f"Error fetching historical average: {e}")
        return None

# Test with Ash Creek for October 17, 2026
print("Testing Historical Weather Average for Ash Creek")
print("=" * 60)
lat = 39.5
lon = -120.5
test_date = datetime(2026, 10, 17)

print(f"\nLocation: Ash Creek ({lat}, {lon})")
print(f"Target Date: {test_date.strftime('%B %d, %Y')}")
print(f"Fetching historical data from 2021-2025...")
print("-" * 60)

weather = fetch_historical_weather_average(lat, lon, test_date)

if weather:
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Temperature: {weather['temp_mean']:.1f}°F (High: {weather['temp_max']:.1f}°F, Low: {weather['temp_min']:.1f}°F)")
    print(f"Wind: {weather['windspeed']:.1f} mph {weather['wind_direction_cardinal']}")
    print(f"Precipitation: {weather['precipitation']:.2f} inches")
    print(f"Historical Average: {weather.get('is_historical_average', False)}")
    print("\n✓ Weather data successfully retrieved!")
else:
    print("\n✗ Failed to retrieve weather data")
