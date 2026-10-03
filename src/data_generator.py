import numpy as np
import pandas as pd
from pathlib import Path


def generate_energy_data(days=90, seed=42):
    """
    Generate synthetic building energy data.

    The generated dataset contains:
    - timestamp
    - temperature
    - solar_generation
    - electricity_price
    - energy_demand
    """

    np.random.seed(seed)

    hours = days * 24

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=hours,
        freq="h"
    )

    hour = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values

    # 1. Temperature
    
    seasonal_temperature = (
        12
        + 8 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
    )

    daily_temperature = (
        3 * np.sin(2 * np.pi * (hour - 6) / 24)
    )

    temperature = (
        seasonal_temperature
        + daily_temperature
        + np.random.normal(0, 1.5, hours)
    )


    # 2. Solar generation

    daylight = np.maximum(
        0,
        np.sin(np.pi * (hour - 6) / 12)
    )

    solar_generation = (
        5
        * daylight
        * (0.8 + 0.2 * np.random.random(hours))
    )

    
    # 3. Electricity price
   
    electricity_price = np.full(hours, 0.15)

    peak_hours = (hour >= 17) & (hour <= 21)
    electricity_price[peak_hours] = 0.30

    night_hours = (hour >= 0) & (hour <= 6)
    electricity_price[night_hours] = 0.10

    electricity_price += np.random.normal(
        0,
        0.01,
        hours
    )

    electricity_price = np.maximum(
        electricity_price,
        0.05
    )

   
    # 4. Building energy demand
    
    base_demand = 2.0

    morning_demand = (
        1.5
        * np.exp(-((hour - 8) ** 2) / 8)
    )

    evening_demand = (
        2.0
        * np.exp(-((hour - 19) ** 2) / 10)
    )

    temperature_effect = (
        0.12 * np.maximum(18 - temperature, 0)
        + 0.15 * np.maximum(temperature - 24, 0)
    )

    noise = np.random.normal(
        0,
        0.2,
        hours
    )

    energy_demand = (
        base_demand
        + morning_demand
        + evening_demand
        + temperature_effect
        + noise
    )

    energy_demand = np.maximum(
        energy_demand,
        0.5
    )

    
    # Create DataFrame

    data = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "solar_generation": solar_generation,
        "electricity_price": electricity_price,
        "energy_demand": energy_demand
    })

    return data


if __name__ == "__main__":

    data = generate_energy_data(
        days=90,
        seed=42
    )

    output_path = Path(
        "data/raw/energy_data.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        output_path,
        index=False
    )

    print("Dataset generated successfully!")
    print(f"Shape: {data.shape}")
    print(f"Saved to: {output_path}")
    print()
    print(data.head())