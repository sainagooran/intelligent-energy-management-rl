import pandas as pd
from pathlib import Path


# Load raw dataset
data = pd.read_csv(
    "data/raw/energy_data.csv",
    parse_dates=["timestamp"]
)


# Chronological split
# First 60 days -> training
# Last 30 days -> testing

split_index = int(len(data) * 0.67)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()


# Create output directory
output_dir = Path("data/processed")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# Save datasets
train_data.to_csv(
    output_dir / "train_data.csv",
    index=False
)

test_data.to_csv(
    output_dir / "test_data.csv",
    index=False
)


print("Data split completed successfully!")

print()
print("Full dataset:")
print(f"Rows: {len(data)}")

print()
print("Training dataset:")
print(f"Rows: {len(train_data)}")
print(
    f"From: {train_data['timestamp'].min()}"
)
print(
    f"To: {train_data['timestamp'].max()}"
)

print()
print("Testing dataset:")
print(f"Rows: {len(test_data)}")
print(
    f"From: {test_data['timestamp'].min()}"
)
print(
    f"To: {test_data['timestamp'].max()}"
)

print()
print("Files saved:")
print("data/processed/train_data.csv")
print("data/processed/test_data.csv")
