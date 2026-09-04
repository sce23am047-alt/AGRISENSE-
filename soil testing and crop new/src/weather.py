"""
src/weather.py
Live weather forecast — fetched from OpenWeatherMap (free tier).

IMPORTANT: This module returns real, live data from an external API.
It is NOT a trained model and does NOT predict anything — it is
explicitly separate from the crop/fertilizer ML pipeline. The UI must
always label this as "Live Forecast", never as an AI prediction.
"""

import requests
import config


def weather_configured() -> bool:
    """True if a WEATHER_API_KEY has been set in .env."""
    return bool(config.WEATHER_API_KEY)


def get_forecast(city: str = None) -> dict:
    """
    Fetch a 5-day / 3-hour-step forecast for the given city (defaults to
    config.WEATHER_DEFAULT_CITY) and collapse it into one entry per day
    (using the midday reading, which is most representative).

    Returns:
        {"city": str, "days": [{"date": "YYYY-MM-DD", "temp": float,
         "condition": str, "icon": str, "humidity": int}, ...]}
        or {"error": str} if the request fails.
    """
    if not weather_configured():
        return {"error": "WEATHER_API_KEY is not set in .env. "
                          "Get a free key at openweathermap.org and add it to .env."}

    city = city or config.WEATHER_DEFAULT_CITY

    try:
        response = requests.get(
            config.WEATHER_API_BASE_URL,
            params={
                "q": city,
                "appid": config.WEATHER_API_KEY,
                "units": "metric",
            },
            timeout=5,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not reach the weather service: {e}"}

    data = response.json()
    if str(data.get("cod")) != "200":
        return {"error": data.get("message", "Unknown error from weather API.")}

    # The API returns readings every 3 hours for 5 days. Keep the one
    # closest to midday (12:00) for each calendar date, so we get a
    # clean 5-day summary instead of 40 raw entries.
    daily = {}
    for entry in data.get("list", []):
        date_str, time_str = entry["dt_txt"].split(" ")
        if date_str not in daily or time_str == "12:00:00":
            daily[date_str] = {
                "date": date_str,
                "temp": round(entry["main"]["temp"], 1),
                "condition": entry["weather"][0]["main"],
                "icon": entry["weather"][0]["icon"],
                "humidity": entry["main"]["humidity"],
            }

    return {
        "city": data.get("city", {}).get("name", city),
        "days": list(daily.values())[:5],
    }