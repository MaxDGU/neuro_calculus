import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import shutil # For cleaning up output directory

from neuro_change_tracker import Tracker, Comparer, Plotter, Snapshot

# 0. Configuration
NUM_EPOCHS = 5
SNAPSHOT_DIR = "full_analysis_output/snapshots"
PLOT_DIR = "full_analysis_output/plots"
CLEANUP_PREVIOUS_OUTPUT = True

# 1. Define a simple model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784) # Flatten input
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    print("Starting Neuro Change Tracker full analysis example...")

    if CLEANUP_PREVIOUS_OUTPUT:
        if os.path.exists("full_analysis_output"):
            print("Cleaning up previous output directory...")
            shutil.rmtree("full_analysis_output")

    # Ensure output directories exist
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 2. Initialize model, optimizer, and loss
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    print("Model, optimizer, and criterion initialized.")

    # 3. Initialize Tracker
    # The Tracker will save snapshots into a timestamped subdirectory inside SNAPSHOT_DIR
    tracker = Tracker(model, save_dir=SNAPSHOT_DIR)
    print(f"Tracker initialized. Snapshots will be saved in subdirectories of: {SNAPSHOT_DIR}")

    # Dummy data for training loop
    dummy_input = torch.randn(64, 784) # Batch of 64, 28x28 flattened images
    dummy_targets = torch.randint(0, 10, (64,)) # Batch of 64 labels

    # --- Training Loop & Snapshotting ---
    print("\n--- Starting Dummy Training Loop ---_tracker.py")
    # Capture initial state
    tracker.capture_snapshot(epoch=-1, step=-1, custom_metadata={"stage": "initialization"})
    print(f"Captured initial snapshot (Epoch -1, Step -1) in {tracker.save_dir}")

    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        optimizer.step()

        # Capture snapshot after each epoch
        tracker.capture_snapshot(epoch=epoch, step=(epoch + 1) * 100, custom_metadata={"loss": loss.item()})
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss.item():.4f}. Snapshot captured in {tracker.save_dir}")
        time.sleep(0.01) # Small delay for distinct timestamps if needed

    print("--- Dummy Training Loop Finished ---")

    # 4. Load Snapshots (if needed, or use tracker.get_all_snapshots() if still in memory)
    # For this example, let's assume we might run comparison as a separate step
    # So we load them from the directory tracker saved them into.
    # tracker.save_dir points to the specific timestamped directory for this run.
    print(f"\n--- Loading Snapshots from {tracker.save_dir} ---")
    # snapshots_list = Tracker.load_snapshots_from_dir(tracker.save_dir)
    # OR, if the tracker object is still available and has them in memory:
    snapshots_list = tracker.get_all_snapshots()
    if not snapshots_list:
        print("Error: No snapshots found or loaded. Exiting.")
        return
    print(f"Loaded {len(snapshots_list)} snapshots.")

    # 5. Initialize Comparer and Perform Comparisons
    comparer = Comparer()
    print("\n--- Performing Comparisons --- ")
    consecutive_comparisons = comparer.compare_consecutive_snapshots(snapshots_list, per_layer=True)
    
    if not consecutive_comparisons:
        print("No consecutive comparisons were generated.")
    else:
        print(f"Generated {len(consecutive_comparisons)} consecutive comparisons.")
        # Print summary of the first comparison as an example
        if consecutive_comparisons:
            first_comp = consecutive_comparisons[0]
            print("\nExample of first comparison result:")
            print(f"  Snapshot 1 Meta: {first_comp['snapshot1_metadata']}")
            print(f"  Snapshot 2 Meta: {first_comp['snapshot2_metadata']}")
            print(f"  Aggregate L2: {first_comp['aggregate_L2_distance_from_layers']:.4f}")
            # print(f"  Per-layer L2: {first_comp['per_layer_L2_distance']}")

    # 6. Initialize Plotter and Generate Plots
    if consecutive_comparisons:
        print("\n--- Generating Plots --- ")
        plotter = Plotter(comparison_results=consecutive_comparisons, output_dir=PLOT_DIR)
        
        plotter.plot_total_change_over_epochs(filename="total_model_change_vs_epoch.png")
        plotter.plot_per_layer_change_over_epochs(filename_prefix="per_layer_model_change")
        print(f"Plots saved in {PLOT_DIR}")
    else:
        print("Skipping plotting as no comparison results are available.")

    print("\nNeuro Change Tracker full analysis example finished.")
    print(f"Snapshots are in: {tracker.save_dir}")
    print(f"Plots are in: {PLOT_DIR}")

if __name__ == "__main__":
    main() 