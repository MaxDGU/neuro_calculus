import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any, Optional

# Assuming comparison results are structured as list of dicts from Comparer
# Each dict would look like:
# {
#     "snapshot1_metadata": snap1.metadata, (e.g., {"epoch": 0, "step": 0})
#     "snapshot2_metadata": snap2.metadata, (e.g., {"epoch": 1, "step": 100})
#     "comparison_type": "L2_distance",
#     "aggregate_L2_distance_from_layers": float_value,
#     "per_layer_L2_distance": {"layer_name": float_value, ...}
# }

class Plotter:
    def __init__(self, comparison_results: List[Dict[str, Any]], output_dir: str = "neuro_plots"):
        self.comparison_results = comparison_results
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_total_change_over_epochs(self, filename: str = "total_change_vs_epoch.png"):
        """Plots the total weight change (aggregate L2 distance) over epochs."""
        if not self.comparison_results:
            print("No comparison results to plot.")
            return

        epochs = []
        total_changes = []

        for comp in self.comparison_results:
            # Assuming comparison is between epoch N and N+1, associate change with epoch N+1
            # or the midpoint, or simply the index of comparison.
            # For simplicity, let's use the epoch of the second snapshot.
            if comp.get("snapshot2_metadata") and comp["snapshot2_metadata"].get("epoch") is not None:
                epochs.append(comp["snapshot2_metadata"]["epoch"])
            else:
                # Fallback if epoch info is missing, use index (less ideal)
                epochs.append(len(epochs))
            
            total_changes.append(comp.get("aggregate_L2_distance_from_layers", 0))

        if not epochs or not total_changes:
            print("Could not extract epochs or total changes for plotting.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, total_changes, marker='o', linestyle='-')
        plt.title('Total Weight Change (Aggregate L2 Distance) Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Aggregate L2 Distance')
        plt.grid(True)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved to {filepath}")

    def plot_per_layer_change_over_epochs(self, filename_prefix: str = "layer_change_vs_epoch"):
        """Plots the L2 distance change for each layer over epochs."""
        if not self.comparison_results:
            print("No comparison results to plot.")
            return

        layer_names = set()
        for comp in self.comparison_results:
            if "per_layer_L2_distance" in comp:
                layer_names.update(comp["per_layer_L2_distance"].keys())
        
        if not layer_names:
            print("No per-layer distance information found.")
            return

        # Prepare data for plotting: {layer_name: {'epochs': [], 'changes': []}}
        plot_data: Dict[str, Dict[str, List[Any]]] = {name: {"epochs": [], "changes": []} for name in layer_names}

        for comp in self.comparison_results:
            epoch = None
            if comp.get("snapshot2_metadata") and comp["snapshot2_metadata"].get("epoch") is not None:
                epoch = comp["snapshot2_metadata"]["epoch"]
            else:
                # If epoch is not found in snapshot2, try snapshot1, or assign a default
                # This part needs careful handling based on how comparisons are structured.
                # For now, we'll skip if epoch is indeterminable for a data point for a layer.
                pass # Fallback to index might be needed if epochs are sparse or irregular

            if "per_layer_L2_distance" in comp:
                for layer_name, dist in comp["per_layer_L2_distance"].items():
                    current_epoch_for_layer = epoch
                    if current_epoch_for_layer is None:
                        # If epoch from metadata is None, use the count of existing data points for this layer as x-axis
                        current_epoch_for_layer = len(plot_data[layer_name]["epochs"])
                    
                    plot_data[layer_name]["epochs"].append(current_epoch_for_layer)
                    plot_data[layer_name]["changes"].append(dist)
        
        num_layers = len(layer_names)
        if num_layers == 0:
            print("No layers to plot.")
            return

        # Create a single figure with subplots for each layer if few, or individual plots if many.
        # For simplicity, let's start with individual plots per layer.

        for layer_name in sorted(list(layer_names)):
            data = plot_data[layer_name]
            if not data["epochs"] or not data["changes"]:
                print(f"No data to plot for layer: {layer_name}")
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(data["epochs"], data["changes"], marker='o', linestyle='-')
            plt.title(f'Weight Change (L2 Distance) for Layer: {layer_name} Over Epochs')
            plt.xlabel('Epoch / Snapshot Index') # Label might need to be more generic if epochs aren't always present
            plt.ylabel('L2 Distance')
            plt.grid(True)
            
            layer_filename = f"{filename_prefix}_{layer_name.replace('.', '_')}.png"
            filepath = os.path.join(self.output_dir, layer_filename)
            plt.savefig(filepath)
            plt.close()
            print(f"Plot saved to {filepath}")

# Example Usage (for testing, to be integrated or called from another script)
if __name__ == '__main__':
    # Dummy comparison results (replace with actual results from Comparer)
    dummy_comparison_results = [
        {
            "snapshot1_metadata": {"epoch": 0, "step": 0, "name": "initial"},
            "snapshot2_metadata": {"epoch": 1, "step": 100, "name": "epoch_1"},
            "comparison_type": "L2_distance",
            "aggregate_L2_distance_from_layers": 0.5,
            "per_layer_L2_distance": {"fc1.weight": 0.3, "fc1.bias": 0.1, "fc2.weight": 0.05, "fc2.bias": 0.05}
        },
        {
            "snapshot1_metadata": {"epoch": 1, "step": 100, "name": "epoch_1"},
            "snapshot2_metadata": {"epoch": 2, "step": 200, "name": "epoch_2"},
            "comparison_type": "L2_distance",
            "aggregate_L2_distance_from_layers": 0.8,
            "per_layer_L2_distance": {"fc1.weight": 0.4, "fc1.bias": 0.15, "fc2.weight": 0.15, "fc2.bias": 0.1}
        },
        {
            "snapshot1_metadata": {"epoch": 2, "step": 200, "name": "epoch_2"},
            "snapshot2_metadata": {"epoch": 3, "step": 300, "name": "epoch_3"},
            "comparison_type": "L2_distance",
            "aggregate_L2_distance_from_layers": 0.3,
            "per_layer_L2_distance": {"fc1.weight": 0.1, "fc1.bias": 0.05, "fc2.weight": 0.1, "fc2.bias": 0.05}
        }
    ]

    print("[INFO] Initializing Plotter with dummy data...")
    plotter = Plotter(comparison_results=dummy_comparison_results, output_dir="test_plots_output")
    
    print("\n[INFO] Plotting total change over epochs...")
    plotter.plot_total_change_over_epochs()
    
    print("\n[INFO] Plotting per-layer change over epochs...")
    plotter.plot_per_layer_change_over_epochs()
    
    print("\n[INFO] Plotting example finished. Check the 'test_plots_output' directory.") 