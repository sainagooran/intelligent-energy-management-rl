import pandas as pd

from energy_environment import EnergyEnvironment


data = pd.read_csv(
    "data/raw/energy_data.csv"
)

env = EnergyEnvironment(data)

observation, info = env.reset()

print("Initial observation:")
print(observation)

print("\nAction: DISCHARGE")

next_observation, reward, terminated, truncated, info = env.step(2)

print("\nNext observation:")
print(next_observation)

print("\nReward:")
print(reward)

print("\nInfo:")
print(info)

print("\nTerminated:")
print(terminated)

print("\nTruncated:")
print(truncated)