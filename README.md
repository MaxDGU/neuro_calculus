# Neuro Change Tracker

A system for tracking, analyzing, and visualizing changes in neural network weights during training. 
This tool aims to be model-agnostic and training-regime-agnostic, providing a 'calculus of learning' for neural networks.

See `PLAN.md` for the project roadmap and future goals.

## Features (Current)

- **Snapshotting**: Capture model `state_dict` at various training points (e.g., per epoch).
- **Comparison**: Calculate L2 distance changes between snapshots, both for the entire model and per layer.
- **Visualization**: Generate plots for total weight change and per-layer weight changes over epochs.
- **Modular Design**: Core components for tracking, comparison, and plotting are separated for flexibility.

## Installation

It is recommended to use a Conda environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MaxDGU/neuro_calculus.git
    cd neuro_calculus
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n neuro_tracker_env python=3.9 -y
    conda activate neuro_tracker_env
    ```
    *(You can replace `neuro_tracker_env` with your preferred environment name and `python=3.9` with another Python version if needed, though 3.9 is tested.)*

3.  **Install dependencies and the package:**
    The `pyproject.toml` file specifies dependencies. Install them and the package in editable mode:
    ```bash
    pip install -e .
    ```
    *(This will also install `torch`, `matplotlib`, and `seaborn` as specified in `pyproject.toml`)*

## Quick Start / Example Usage

The script `run_analysis_example.py` in the root directory demonstrates a full workflow:

1.  **Run the example:**
    Make sure your Conda environment is activated (`conda activate neuro_tracker_env`).
    ```bash
    python run_analysis_example.py
    ```

2.  **Expected Output:**
    - The script will print progress to the console, including loss values during a dummy training loop, paths to saved snapshots, and paths to generated plots.
    - Snapshots will be saved in a timestamped subdirectory within `full_analysis_output/snapshots/`.
    - Plots visualizing weight changes will be saved in `full_analysis_output/plots/`.

**Understanding the Example (`run_analysis_example.py`):**

-   **Initialization**: A simple MLP model is defined. `Tracker`, `Comparer`, and `Plotter` objects are initialized.
-   **Tracking**: During a dummy training loop, `tracker.capture_snapshot(...)` is called to save the model's state dictionary along with metadata like epoch and loss.
-   **Comparison**: After training, `comparer.compare_consecutive_snapshots(...)` is used to calculate the L2 distance between the weights of consecutive snapshots.
-   **Visualization**: `plotter.plot_total_change_over_epochs(...)` and `plotter.plot_per_layer_change_over_epochs(...)` generate and save PNG images of the weight changes.

## Core Components

-   `neuro_change_tracker.core.Snapshot`: Represents a single snapshot of a model's state and associated metadata.
-   `neuro_change_tracker.core.Tracker`: Manages the capturing and saving of `Snapshot` objects from a PyTorch model during training.
-   `neuro_change_tracker.comparison.Comparer`: Provides methods to calculate differences (e.g., L2 distance) between `Snapshot` objects, either per-layer or for the total model.
-   `neuro_change_tracker.visualization.Plotter`: Takes comparison results and generates plots using Matplotlib/Seaborn.

## How to Use in Your Own Project

1.  **Import necessary classes:**
    ```python
    from neuro_change_tracker import Tracker, Comparer, Plotter
    import torch
    # Define or load your PyTorch model
    # model = YourCustomModel()
    ```

2.  **Initialize Tracker:**
    ```python
    # model = YourPyTorchModel()
    # tracker = Tracker(model, save_dir="my_experiment_snapshots")
    ```

3.  **Capture Snapshots during your training loop:**
    ```python
    # For epoch in range(num_epochs):
    #     # ... your training code ...
    #     tracker.capture_snapshot(epoch=epoch, custom_metadata={"your_metric": value})
    ```

4.  **Perform Comparison:**
    ```python
    # snapshots = tracker.get_all_snapshots() # Or load from disk: Tracker.load_snapshots_from_dir(tracker.save_dir)
    # comparer = Comparer()
    # comparison_results = comparer.compare_consecutive_snapshots(snapshots)
    ```

5.  **Generate Plots:**
    ```python
    # if comparison_results:
    #     plotter = Plotter(comparison_results, output_dir="my_experiment_plots")
    #     plotter.plot_total_change_over_epochs()
    #     plotter.plot_per_layer_change_over_epochs()
    ```

## Contributing

(Contribution guidelines to be added - see `PLAN.md`)

## License

(License to be confirmed - currently placeholder MIT in `pyproject.toml` - see `PLAN.md`)

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