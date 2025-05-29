# Neuro Change Tracker

A system for tracking changes in neural network weights during training.

## Installation

```bash
pip install -r requirements.txt
# For editable install:
# pip install -e .
```

## Usage

(More details to come)

```python
from neuro_change_tracker import Tracker
import torch

# Example model
class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP()

# Initialize tracker
# Snapshots will be saved in a timestamped subdirectory within 'my_experiment_snapshots'
tracker = Tracker(model, save_dir="my_experiment_snapshots")

# Simulate training loop
num_epochs = 3
for epoch in range(num_epochs):
    # ... your training step ...
    # model weights would be updated here

    # Capture snapshot
    tracker.capture_snapshot(epoch=epoch, step=epoch*100) # Example step, adjust as needed
    print(f"Captured snapshot for epoch {epoch}")

print(f"All snapshots saved in: {tracker.save_dir}")
```

See `PLAN.md` for the project roadmap. 