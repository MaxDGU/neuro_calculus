import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any

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
    """
    Generates plots to visualize weight changes from comparison results.

    The plotter takes a list of comparison result dictionaries, typically generated
    by the `Comparer` class, and produces plots for total and per-layer weight 
    changes over epochs (or snapshot indices).

    Expected structure for each dict in `comparison_results`:
    {
        "snapshot1_metadata": { "epoch": int, ... },
        "snapshot2_metadata": { "epoch": int, ... },
        "aggregate_L2_distance_from_layers": float,
        "per_layer_L2_distance": { "layer_name1": float, "layer_name2": float, ... }
        ...
    }
    """
    def __init__(self, comparison_results: List[Dict[str, Any]], output_dir: str = "neuro_plots"):
        """
        Initializes a Plotter object.

        Args:
            comparison_results (List[Dict[str, Any]]): A list of dictionaries, where 
                each dictionary contains the results of comparing two snapshots.
            output_dir (str, optional): The directory where generated plots will be 
                                      saved. Defaults to "neuro_plots".
        """
        self.comparison_results = comparison_results
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid") # Apply a seaborn theme for aesthetics

    def plot_total_change_over_epochs(self, filename: str = "total_change_vs_epoch.png"):
        """
        Plots the total weight change (aggregate L2 distance) over epochs.

        The x-axis represents the epoch of the second snapshot in each comparison pair.
        If epoch information is missing, it falls back to using the comparison index.

        Args:
            filename (str, optional): The name for the output plot file. 
                                      Defaults to "total_change_vs_epoch.png".
        """
        if not self.comparison_results:
            print("[Plotter] No comparison results to plot for total change.")
            return

        epochs = []
        total_changes = []

        for i, comp in enumerate(self.comparison_results):
            epoch_val = None
            if comp.get("snapshot2_metadata") and comp["snapshot2_metadata"].get("epoch") is not None:
                epoch_val = comp["snapshot2_metadata"]["epoch"]
            else:
                epoch_val = i # Fallback to index if epoch is not available
            epochs.append(epoch_val)
            
            change_val = comp.get("aggregate_L2_distance_from_layers")
            if change_val is None or change_val < 0: # Handle missing or invalid (-1) aggregate distances
                print(f"[Plotter] Warning: Missing or invalid aggregate change for comparison index {i}. Using 0 for plot.")
                total_changes.append(0) # Or np.nan if preferred, but 0 is simpler for basic plot
            else:
                total_changes.append(change_val)

        if not epochs:
            print("[Plotter] Could not extract sufficient data for plotting total change.")
            return

        plt.figure(figsize=(12, 7))
        plt.plot(epochs, total_changes, marker='o', linestyle='-', color='b')
        plt.title('Total Model Weight Change (Aggregate L2 Distance) Over Training Progression', fontsize=16)
        plt.xlabel('Epoch / Snapshot Index of Second Snapshot in Pair', fontsize=12)
        plt.ylabel('Aggregate L2 Distance', fontsize=12)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"[Plotter] Plot saved: {filepath}")

    def plot_per_layer_change_over_epochs(self, filename_prefix: str = "layer_change_vs_epoch"):
        """
        Plots the L2 distance change for each layer over epochs (or snapshot indices).

        Generates a separate plot for each layer found in the comparison results.
        The x-axis uses epoch data if available from the second snapshot in a pair,
        otherwise falls back to an index.

        Args:
            filename_prefix (str, optional): The prefix for the output plot filenames. 
                                             Each plot will be named 
                                             `{filename_prefix}_{layer_name}.png`. 
                                             Defaults to "layer_change_vs_epoch".
        """
        if not self.comparison_results:
            print("[Plotter] No comparison results to plot for per-layer changes.")
            return

        layer_names = set()
        for comp in self.comparison_results:
            if "per_layer_L2_distance" in comp and isinstance(comp["per_layer_L2_distance"], dict):
                layer_names.update(comp["per_layer_L2_distance"].keys())
        
        if not layer_names:
            print("[Plotter] No per-layer distance information found or layers are not in dict format.")
            return

        plot_data: Dict[str, Dict[str, List[Any]]] = {name: {"x_axis_vals": [], "changes": []} for name in layer_names}

        for i, comp in enumerate(self.comparison_results):
            x_val = None
            if comp.get("snapshot2_metadata") and comp["snapshot2_metadata"].get("epoch") is not None:
                x_val = comp["snapshot2_metadata"]["epoch"]
            else:
                x_val = i # Fallback to comparison index

            if "per_layer_L2_distance" in comp and isinstance(comp["per_layer_L2_distance"], dict):
                for layer_name, dist in comp["per_layer_L2_distance"].items():
                    if layer_name in plot_data: # Ensure layer was identified initially
                        plot_data[layer_name]["x_axis_vals"].append(x_val)
                        if dist is None or dist < 0: # Handle missing or invalid (-1) distances
                            print(f"[Plotter] Warning: Missing or invalid distance for layer '{layer_name}' at index {i}. Using 0 for plot.")
                            plot_data[layer_name]["changes"].append(0)
                        else:
                            plot_data[layer_name]["changes"].append(dist)
        
        if not any(plot_data[name]["x_axis_vals"] for name in layer_names):
            print("[Plotter] No valid data points found to plot for any layer.")
            return

        for layer_name in sorted(list(layer_names)):
            data = plot_data[layer_name]
            if not data["x_axis_vals"] or not data["changes"]:
                # This check might be redundant due to the one above, but good for safety
                print(f"[Plotter] No data to plot for layer: {layer_name}")
                continue

            plt.figure(figsize=(12, 7))
            plt.plot(data["x_axis_vals"], data["changes"], marker='o', linestyle='-')
            plt.title(f'Weight Change (L2 Distance) for Layer: {layer_name}', fontsize=16)
            plt.xlabel('Epoch / Snapshot Index of Second Snapshot in Pair', fontsize=12)
            plt.ylabel('L2 Distance', fontsize=12)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            
            # Sanitize layer_name for filename
            safe_layer_name = layer_name.replace('.', '_').replace('[', '_').replace(']', '')
            layer_filename = f"{filename_prefix}_{safe_layer_name}.png"
            filepath = os.path.join(self.output_dir, layer_filename)
            plt.savefig(filepath)
            plt.close()
            print(f"[Plotter] Plot saved: {filepath}")

# Example Usage (primary example is run_analysis_example.py)
if __name__ == '__main__':
    # Dummy comparison results
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