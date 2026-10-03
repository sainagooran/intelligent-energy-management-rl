import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EnergyEnvironment(gym.Env):

    def __init__(
        self,
        data,
        battery_capacity=10.0,
        max_charge_rate=3.0,
        max_discharge_rate=3.0,
        initial_battery=5.0
    ):
        super().__init__()

        self.data = data.reset_index(drop=True)

        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.initial_battery = initial_battery

        # 0 = Normal / grid
        # 1 = Charge battery
        # 2 = Discharge battery
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([
                -50.0,
                0.0,
                0.0,
                0.0,
                0.0
            ], dtype=np.float32),

            high=np.array([
                60.0,
                10.0,
                1.0,
                20.0,
                battery_capacity
            ], dtype=np.float32)
        )

        self.current_step = 0
        self.battery_level = initial_battery

    def _get_state(self):

        row = self.data.iloc[self.current_step]

        return np.array([
            row["temperature"],
            row["solar_generation"],
            row["electricity_price"],
            row["energy_demand"],
            self.battery_level
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0
        self.battery_level = self.initial_battery

        return self._get_state(), {}

    def step(self, action):

        row = self.data.iloc[self.current_step]

        solar = row["solar_generation"]
        demand = row["energy_demand"]
        price = row["electricity_price"]

        # Energy after solar generation
        net_demand = demand - solar

        grid_energy = 0.0
        charge_amount = 0.0
        discharge_amount = 0.0

        
        # Action 0: Normal operation

        if action == 0:

            grid_energy = max(
                net_demand,
                0
            )

    
        # Action 1: Charge battery
       
        elif action == 1:

            # First use solar surplus
            solar_surplus = max(
                solar - demand,
                0
            )

            solar_charge = min(
                solar_surplus,
                self.max_charge_rate,
                self.battery_capacity - self.battery_level
            )

            self.battery_level += solar_charge
            charge_amount += solar_charge

            remaining_charge_capacity = min(
                self.max_charge_rate - solar_charge,
                self.battery_capacity - self.battery_level
            )

            # If capacity remains, buy electricity from grid
            if remaining_charge_capacity > 0:

                grid_charge = remaining_charge_capacity

                self.battery_level += grid_charge
                charge_amount += grid_charge

                grid_energy = max(
                    net_demand,
                    0
                ) + grid_charge

            else:

                grid_energy = max(
                    net_demand,
                    0
                )

    
        # Action 2: Discharge battery
    
        elif action == 2:

            discharge_amount = min(
                self.max_discharge_rate,
                self.battery_level,
                max(net_demand, 0)
            )

            self.battery_level -= discharge_amount

            grid_energy = max(
                net_demand - discharge_amount,
                0
            )


        # Cost
      
        cost = grid_energy * price

        # Reward = negative cost
        reward = -cost

        # Move to next timestep

        self.current_step += 1

        terminated = (
            self.current_step >= len(self.data) - 1
        )

        truncated = False

        if not terminated:

            next_observation = self._get_state()

        else:

            next_observation = np.zeros(
                5,
                dtype=np.float32
            )

        info = {
            "grid_energy": grid_energy,
            "battery_level": self.battery_level,
            "cost": cost,
            "charge_amount": charge_amount,
            "discharge_amount": discharge_amount
        }

        return (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        )