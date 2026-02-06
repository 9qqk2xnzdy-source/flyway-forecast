import requests
import pandas as pd
from datetime import datetime

# Test weather API for Ash Creek
lat = 39.5
lon = -120.5
test_date = datetime(2026, 10, 17)  # October 17, 2026
date_str = test_date.strftime('%Y-%m-%d')

print(f"Testing weather API for Ash Creek")
print(f"Coordinates: {lat}, {lon}")
print(f"Date: {date_str}")
print("-" * 50)

# Parameters for comprehensive weather data
params = (
    "daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
    "precipitation_sum,windspeed_10m_max,winddirection_10m_dominant,"
    "sunrise,sunset,weathercode&"
    "temperature_unit=fahrenheit&windspeed_unit=mph&"
    "precipitation_unit=inch&timezone=America/Los_Angeles"
)

# Test forecast endpoint (for future dates)
url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&{params}"

print(f"\nTesting URL:")
print(url)
print("-" * 50)

try:
    resp = requests.get(url, timeout=15)
    print(f"\nStatus Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print(f"\nResponse JSON:")
        print(data)

        daily = data.get('daily', {})
        times = daily.get('time', [])

        if times:
            print(f"\n✓ Weather data retrieved successfully!")
            print(f"Temperature Max: {daily.get('temperature_2m_max', [None])[0]}")
            print(f"Temperature Min: {daily.get('temperature_2m_min', [None])[0]}")
            print(f"Wind Speed: {daily.get('windspeed_10m_max', [None])[0]}")
        else:
            print(f"\n✗ No time data in response")
    else:
        print(f"\n✗ API returned error status: {resp.status_code}")
        print(f"Response: {resp.text}")

except Exception as e:
    print(f"\n✗ Error: {e}")

# Test with today's date for comparison
print("\n" + "=" * 50)
print("Testing with today's date for comparison:")
print("=" * 50)

today_str = datetime.today().strftime('%Y-%m-%d')
url_today = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={today_str}&end_date={today_str}&{params}"

print(f"\nURL: {url_today}")

try:
    resp = requests.get(url_today, timeout=15)
    print(f"Status Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        daily = data.get('daily', {})
        times = daily.get('time', [])

        if times:
            print(f"✓ Today's weather retrieved successfully!")
            print(f"Temperature Max: {daily.get('temperature_2m_max', [None])[0]}")
        else:
            print(f"✗ No time data in response")
    else:
        print(f"✗ API returned error: {resp.text}")

except Exception as e:
    print(f"✗ Error: {e}")
