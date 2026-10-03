import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from energy_environment import EnergyEnvironment


data = pd.read_csv(
    "data/processed/train_data.csv"
)

env = EnergyEnvironment(data)

print("Checking environment...")

check_env(
    env,
    warn=True
)

print("Environment check completed!")


model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1
)

print("\nStarting DQN training...")

model.learn(
    total_timesteps=50000
)

model.save(
    "results/dqn_energy_management"
)

print("\nTraining completed!")
print("Model trained on train_data.csv")
print("Model saved to results/dqn_energy_management")