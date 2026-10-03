import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(
    "data/raw/energy_data.csv",
    parse_dates=["timestamp"]
)


plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["energy_demand"],
    label="Energy Demand"
)

plt.plot(
    data["timestamp"],
    data["solar_generation"],
    label="Solar Generation"
)

plt.xlabel("Time")
plt.ylabel("Energy (kWh)")
plt.title("Energy Demand and Solar Generation")

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/energy_demand_solar.png",
    dpi=300
)

plt.show()
