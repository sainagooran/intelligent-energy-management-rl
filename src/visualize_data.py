import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(
    "data/raw/energy_data.csv",
    parse_dates=["timestamp"]
)


plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["energy_demand"]
)

plt.title("Building Energy Demand")
plt.xlabel("Time")
plt.ylabel("Energy Demand (kWh)")

plt.tight_layout()
plt.show()