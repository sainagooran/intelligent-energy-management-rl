import matplotlib.pyplot as plt


dqn_cost = 225.60
rule_cost = 197.45

dqn_grid = 1207.45
rule_grid = 1355.12


methods = [
    "DQN",
    "Rule-Based"
]


costs = [
    dqn_cost,
    rule_cost
]


grid_energy = [
    dqn_grid,
    rule_grid
]


plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    costs
)

plt.ylabel("Electricity Cost")
plt.title("Test Electricity Cost Comparison")

plt.tight_layout()

plt.savefig(
    "results/figures/electricity_cost_comparison.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    grid_energy
)

plt.ylabel("Grid Energy (kWh)")
plt.title("Test Grid Energy Comparison")

plt.tight_layout()

plt.savefig(
    "results/figures/grid_energy_comparison.png",
    dpi=300
)

plt.show()
