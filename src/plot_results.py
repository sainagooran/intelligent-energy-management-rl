import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from energy_environment import EnergyEnvironment


data = pd.read_csv(
    "data/processed/test_data.csv"
)

env = EnergyEnvironment(data)

model = DQN.load(
    "results/dqn_energy_management"
)

observation, info = env.reset()

battery_levels = []
timestamps = []

done = False

while not done:

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    observation, reward, terminated, truncated, info = env.step(
        int(action)
    )

    battery_levels.append(
        info["battery_level"]
    )

    timestamps.append(
        data.iloc[len(battery_levels) - 1]["timestamp"]
    )

    done = terminated or truncated


plt.figure(figsize=(12, 5))

plt.plot(
    timestamps,
    battery_levels
)

plt.xlabel("Time")
plt.ylabel("Battery Level (kWh)")
plt.title("Battery Level - DQN")

plt.tight_layout()

plt.savefig(
    "results/figures/battery_dqn.png",
    dpi=300
)

plt.show()
