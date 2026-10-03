import pandas as pd

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

total_reward = 0.0
total_cost = 0.0
total_grid_energy = 0.0

battery_levels = []
actions = []

done = False

while not done:

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    observation, reward, terminated, truncated, info = env.step(
        int(action)
    )

    total_reward += reward
    total_cost += info["cost"]
    total_grid_energy += info["grid_energy"]

    battery_levels.append(
        info["battery_level"]
    )

    actions.append(
        int(action)
    )

    done = terminated or truncated


print("\n========== DQN TEST RESULTS ==========")

print(f"Total reward: {total_reward:.2f}")
print(f"Total electricity cost: {total_cost:.2f}")
print(f"Total grid energy: {total_grid_energy:.2f}")

print(
    f"Final battery level: "
    f"{battery_levels[-1]:.2f} kWh"
)

print("\nAction counts:")
print(f"Grid: {actions.count(0)}")
print(f"Charge: {actions.count(1)}")
print(f"Discharge: {actions.count(2)}")