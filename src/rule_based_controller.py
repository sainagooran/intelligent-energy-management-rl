import pandas as pd

from energy_environment import EnergyEnvironment


data = pd.read_csv(
    "data/processed/test_data.csv"
)

env = EnergyEnvironment(data)

observation, info = env.reset()

total_cost = 0.0
total_grid_energy = 0.0

actions = []

done = False

while not done:

    solar = observation[1]
    price = observation[2]
    demand = observation[3]
    battery = observation[4]

    action = 0

    if solar > demand and battery < env.battery_capacity:
        action = 1

    elif price < 0.12 and battery < env.battery_capacity:
        action = 1

    elif price >= 0.25 and battery > 0:
        action = 2

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    total_cost += info["cost"]
    total_grid_energy += info["grid_energy"]

    actions.append(action)

    done = terminated or truncated


print("\n========== RULE-BASED TEST RESULTS ==========")

print(f"Total electricity cost: {total_cost:.2f}")
print(f"Total grid energy: {total_grid_energy:.2f}")

print(
    f"Final battery level: "
    f"{info['battery_level']:.2f} kWh"
)

print("\nAction counts:")
print(f"Grid: {actions.count(0)}")
print(f"Charge: {actions.count(1)}")
print(f"Discharge: {actions.count(2)}")