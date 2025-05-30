import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import shutil # For cleaning up output directory
import torch.nn.functional as F # For potential aggregate functions

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
    print("\n--- Starting Dummy Training Loop ---")
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
    if not snapshots_list or len(snapshots_list) < 2:
        print("Error: Not enough snapshots found or loaded for detailed analysis. Exiting.")
        return
    print(f"Loaded {len(snapshots_list)} snapshots.")

    # 5. Initialize Comparer and Perform Comparisons
    comparer = Comparer()
    print("\n--- Performing Comparisons (L2 distance plots) --- ")
    consecutive_comparisons = comparer.compare_consecutive_snapshots(snapshots_list, per_layer=True)
    
    if not consecutive_comparisons:
        print("No consecutive comparisons were generated.")
    else:
        print(f"Generated {len(consecutive_comparisons)} consecutive comparisons for L2 distance plots.")
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
        
        plotter.plot_total_change_over_epochs(filename="total_model_L2_change_vs_epoch.png")
        plotter.plot_per_layer_change_over_epochs(filename_prefix="per_layer_model_L2_change")
        print(f"Plots saved in {PLOT_DIR}")
    else:
        print("Skipping L2 distance plots as no comparison results are available.")

    # --- New M10 Plots --- 
    print("\n--- Generating M10 Trajectory and Heatmap Plots --- ")

    # Ensure plotter is initialized. If L2 distance plots were skipped, initialize it here.
    if 'plotter' not in locals() and snapshots_list:
        print("Initializing Plotter for trajectory and heatmap plots...")
        plotter = Plotter(output_dir=PLOT_DIR) # No comparison_results needed if only doing these plots
    elif not snapshots_list:
        print("Cannot generate M10 plots: No snapshots available.")
        return

    # 1. Specific Weight Trajectories
    weights_to_track = {
        "fc1.weight": [(0,0), (0,1), (10,5)], # (row, col) indices
        "fc3.bias": [(0,)] # (index,) for bias
    }
    plotter.plot_specific_weight_trajectories(
        snapshots=snapshots_list, 
        weights_to_plot=weights_to_track,
        filename_prefix="specific_weights"
    )

    # 2. Layer Aggregate Trajectories
    aggregate_functions = {
        "L2_norm": lambda t: t.norm(p=2).item(),
        "mean_abs_val": lambda t: t.abs().mean().item(),
        "std_dev": lambda t: t.std().item()
    }
    plotter.plot_layer_aggregate_trajectories(
        snapshots=snapshots_list,
        aggregate_fns=aggregate_functions,
        # layer_filter=["fc1.weight", "fc3.bias"], # Optional: filter specific layers
        filename_prefix="layer_aggregates"
    )

    # 3. Layer Difference Heatmap - Now with modes
    if len(snapshots_list) >= 2:
        initial_snapshot = snapshots_list[0]
        final_snapshot = snapshots_list[-1]
        s1_id = f"e{initial_snapshot.metadata.get('epoch','i')}_s{initial_snapshot.metadata.get('step','i')}"
        s2_id = f"e{final_snapshot.metadata.get('epoch','f')}_s{final_snapshot.metadata.get('step','f')}"

        layers_for_heatmap = ["fc1.weight", "fc2.bias"]
        heatmap_modes = ["raw", "percentage", "standardized"]

        for layer_name_hm in layers_for_heatmap:
            for mode in heatmap_modes:
                print(f"Generating heatmap for {layer_name_hm} (mode: {mode}) between {s1_id} and {s2_id}")
                plotter.plot_layer_difference_heatmap(
                    initial_snapshot, 
                    final_snapshot, 
                    layer_name=layer_name_hm,
                    mode=mode,
                    robust_colormap=True, # Default, but can be set to False to test
                    filename=f"heatmap_{layer_name_hm.replace('.', '_')}_{mode}_{s1_id}_vs_{s2_id}.png"
                )
            # Example of non-robust colormap for one mode/layer
            if layer_name_hm == "fc1.weight":
                print(f"Generating non-robust heatmap for {layer_name_hm} (mode: raw) between {s1_id} and {s2_id}")
                plotter.plot_layer_difference_heatmap(
                    initial_snapshot, 
                    final_snapshot, 
                    layer_name=layer_name_hm,
                    mode="raw",
                    robust_colormap=False, 
                    filename=f"heatmap_{layer_name_hm.replace('.', '_')}_raw_nonrobust_{s1_id}_vs_{s2_id}.png"
                )

    print("\nNeuro Change Tracker full analysis example finished.")
    print(f"Snapshots are in: {tracker.save_dir}")
    print(f"Plots are in: {PLOT_DIR}")

if __name__ == "__main__":
    main() 