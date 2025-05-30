import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from typing import List, Dict, Any, Callable, Tuple, Optional

from ..core.snapshot import Snapshot

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
    Generates plots to visualize weight changes from comparison results and snapshot data.

    The plotter can take:
    1. A list of comparison result dictionaries (typically from `Comparer`) for plots 
       like total/per-layer change L2 distance over epochs.
    2. A list of `Snapshot` objects directly for plots like weight trajectories.

    Expected structure for each dict in `comparison_results` (for existing plots):
    {
        "snapshot1_metadata": { "epoch": int, ... },
        "snapshot2_metadata": { "epoch": int, ... },
        "aggregate_L2_distance_from_layers": float,
        "per_layer_L2_distance": { "layer_name1": float, "layer_name2": float, ... }
        ...
    }
    """
    def __init__(self, comparison_results: Optional[List[Dict[str, Any]]] = None, output_dir: str = "neuro_plots"):
        """
        Initializes a Plotter object.

        Args:
            comparison_results (Optional[List[Dict[str, Any]]], optional): 
                A list of dictionaries, where each dictionary contains the results 
                of comparing two snapshots. Required for distance-based plots. 
                Defaults to None.
            output_dir (str, optional): The directory where generated plots will be 
                                      saved. Defaults to "neuro_plots".
        """
        self.comparison_results = comparison_results if comparison_results is not None else []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

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
                print(f"[Plotter] No data to plot for layer: {layer_name}")
                continue

            plt.figure(figsize=(12, 7))
            plt.plot(data["x_axis_vals"], data["changes"], marker='o', linestyle='-')
            plt.title(f'Weight Change (L2 Distance) for Layer: {layer_name}', fontsize=16)
            plt.xlabel('Epoch / Snapshot Index of Second Snapshot in Pair', fontsize=12)
            plt.ylabel('L2 Distance', fontsize=12)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            
            safe_layer_name = layer_name.replace('.', '_').replace('[', '_').replace(']', '')
            layer_filename = f"{filename_prefix}_{safe_layer_name}.png"
            filepath = os.path.join(self.output_dir, layer_filename)
            plt.savefig(filepath)
            plt.close()
            print(f"[Plotter] Plot saved: {filepath}")

    def plot_specific_weight_trajectories(self, 
                                          snapshots: List[Snapshot], 
                                          weights_to_plot: Dict[str, List[Tuple[Any, ...]]], 
                                          filename_prefix: str = "specific_weight_traj"):
        """
        Plots the trajectory of specific indexed weights from specified layers over time.

        Args:
            snapshots (List[Snapshot]): List of Snapshot objects, ordered by time/step.
            weights_to_plot (Dict[str, List[Tuple[Any, ...]]]): 
                A dictionary mapping layer names to a list of index tuples. 
                E.g., {"fc1.weight": [(0,0), (0,1)], "fc2.bias": [(0,)]}
            filename_prefix (str, optional): Prefix for the output plot files. 
                                           Defaults to "specific_weight_traj".
        """
        if not snapshots:
            print("[Plotter] No snapshots provided for weight trajectories.")
            return
        if not weights_to_plot:
            print("[Plotter] No specific weights designated for plotting.")
            return

        # Prepare data: { (layer_name, weight_index_tuple_str): {"x_axis_vals": [], "values": []} }
        plot_data: Dict[str, Dict[str, List[Any]]] = {}
        
        all_layer_names = set(weights_to_plot.keys())

        for i, snap in enumerate(snapshots):
            x_val = snap.metadata.get("epoch", i) # Prioritize epoch, then step, then index
            if snap.metadata.get("step") is not None and x_val == i: # If epoch wasn't there, use step
                 x_val = snap.metadata.get("step")
                 x_axis_label = "Step / Snapshot Index"
            else:
                 x_axis_label = "Epoch / Snapshot Index"

            for layer_name, indices_list in weights_to_plot.items():
                if layer_name not in snap.model_state:
                    # print(f"[Plotter] Warning: Layer '{layer_name}' not found in snapshot {i}. Skipping for this snapshot.")
                    continue
                
                layer_tensor = snap.model_state[layer_name]
                for index_tuple in indices_list:
                    try:
                        weight_val = layer_tensor[index_tuple].item()
                        dict_key = f"{layer_name}_idx_{str(index_tuple).replace(',', '_').replace(' ', '')}"

                        if dict_key not in plot_data:
                            plot_data[dict_key] = {"x_axis_vals": [], "values": [], "layer": layer_name, "index": str(index_tuple)}
                        
                        plot_data[dict_key]["x_axis_vals"].append(x_val)
                        plot_data[dict_key]["values"].append(weight_val)
                    except IndexError:
                        print(f"[Plotter] Warning: Index {index_tuple} out of bounds for layer '{layer_name}' in snapshot {i}.")
                    except Exception as e:
                        print(f"[Plotter] Error accessing weight {layer_name}{index_tuple} in snapshot {i}: {e}")

        if not plot_data:
            print("[Plotter] No valid data collected for specific weight trajectories.")
            return

        for key, data in plot_data.items():
            if not data["x_axis_vals"] or not data["values"]:
                continue
            
            plt.figure(figsize=(12, 7))
            plt.plot(data["x_axis_vals"], data["values"], marker='o', linestyle='-')
            plt.title(f'Trajectory for Weight: {data["layer"]}{data["index"]}', fontsize=16)
            plt.xlabel(x_axis_label, fontsize=12)
            plt.ylabel('Weight Value', fontsize=12)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            
            plot_filename = f"{filename_prefix}_{key}.png"
            filepath = os.path.join(self.output_dir, plot_filename)
            plt.savefig(filepath)
            plt.close()
            print(f"[Plotter] Plot saved: {filepath}")

    def plot_layer_aggregate_trajectories(self, 
                                          snapshots: List[Snapshot], 
                                          aggregate_fns: Dict[str, Callable[[torch.Tensor], float]],
                                          layer_filter: Optional[List[str]] = None, 
                                          filename_prefix: str = "layer_aggregate_traj"):
        """
        Plots trajectories of aggregate statistics for layers over time.

        Args:
            snapshots (List[Snapshot]): List of Snapshot objects.
            aggregate_fns (Dict[str, Callable[[torch.Tensor], float]]):
                Dictionary mapping statistic names to functions that compute them from a tensor.
                E.g., {"L2_norm": lambda t: t.norm().item(), "mean_abs": lambda t: t.abs().mean().item()}
            layer_filter (Optional[List[str]], optional): List of layer names to include. 
                                                        If None, all layers are processed. Defaults to None.
            filename_prefix (str, optional): Prefix for output plot files. 
                                           Defaults to "layer_aggregate_traj".
        """
        if not snapshots:
            print("[Plotter] No snapshots provided for aggregate trajectories.")
            return
        if not aggregate_fns:
            print("[Plotter] No aggregate functions provided.")
            return

        # Data: { agg_fn_name: { layer_name: {"x_axis_vals": [], "values": []} } }
        plot_data: Dict[str, Dict[str, Dict[str, List[Any]]]] = {fn_name: {} for fn_name in aggregate_fns.keys()}
        processed_layers = set()

        x_axis_label = "Epoch / Snapshot Index"
        for i, snap in enumerate(snapshots):
            x_val = snap.metadata.get("epoch", i)
            if snap.metadata.get("step") is not None and x_val == i:
                x_val = snap.metadata.get("step")
                x_axis_label = "Step / Snapshot Index"
            
            for layer_name, layer_tensor in snap.model_state.items():
                if layer_filter and layer_name not in layer_filter:
                    continue
                processed_layers.add(layer_name)

                for fn_name, agg_fn in aggregate_fns.items():
                    try:
                        stat_val = agg_fn(layer_tensor)
                        if layer_name not in plot_data[fn_name]:
                            plot_data[fn_name][layer_name] = {"x_axis_vals": [], "values": []}
                        
                        plot_data[fn_name][layer_name]["x_axis_vals"].append(x_val)
                        plot_data[fn_name][layer_name]["values"].append(stat_val)
                    except Exception as e:
                        print(f"[Plotter] Error computing aggregate '{fn_name}' for layer '{layer_name}' in snapshot {i}: {e}")

        if not processed_layers:
            print("[Plotter] No layers processed for aggregate trajectories (check filter or snapshot content).")
            return

        for fn_name, layer_data_map in plot_data.items():
            plt.figure(figsize=(12, 7))
            for layer_name, data in sorted(layer_data_map.items()):
                if data["x_axis_vals"] and data["values"]:
                    plt.plot(data["x_axis_vals"], data["values"], marker='o', linestyle='-', label=layer_name)
            
            plt.title(f'Trajectory of Layer Aggregate: {fn_name}', fontsize=16)
            plt.xlabel(x_axis_label, fontsize=12)
            plt.ylabel('Aggregate Value', fontsize=12)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            
            plot_filename = f"{filename_prefix}_{fn_name}.png"
            filepath = os.path.join(self.output_dir, plot_filename)
            plt.savefig(filepath)
            plt.close()
            print(f"[Plotter] Plot saved: {filepath}")

    def plot_layer_difference_heatmap(self, 
                                      snapshot1: Snapshot, 
                                      snapshot2: Snapshot, 
                                      layer_name: str, 
                                      filename: Optional[str] = None):
        """
        Plots a heatmap of the difference between a specific layer's weights from two snapshots.

        Args:
            snapshot1 (Snapshot): The first snapshot.
            snapshot2 (Snapshot): The second snapshot.
            layer_name (str): The name of the layer to compare (e.g., 'fc1.weight').
            filename (Optional[str], optional): Name for the output plot file. 
                                                If None, a default is generated. 
                                                Defaults to None.
        """
        if layer_name not in snapshot1.model_state or layer_name not in snapshot2.model_state:
            print(f"[Plotter] Layer '{layer_name}' not found in one or both snapshots. Cannot plot heatmap.")
            return

        w1 = snapshot1.model_state[layer_name]
        w2 = snapshot2.model_state[layer_name]

        if w1.shape != w2.shape:
            print(f"[Plotter] Shapes of layer '{layer_name}' differ between snapshots: {w1.shape} vs {w2.shape}. Cannot plot heatmap.")
            return
        
        diff = w2 - w1
        if diff.ndim == 1: # For bias vectors, reshape to 2D for heatmap
            diff = diff.unsqueeze(0)
        elif diff.ndim == 0: # Scalar tensor
             print(f"[Plotter] Layer '{layer_name}' is a scalar. Heatmap not suitable.")
             return
        elif diff.ndim > 2:
            # For >2D tensors (e.g., conv filters), take a slice or norm along other dims?
            # For now, just warn and skip, or try to flatten last two dims if appropriate.
            # Let's try to reduce to 2D by taking L2 norm across other dimensions if it's a 4D (common for Conv2D)
            if diff.ndim == 4: # Typical (out_channels, in_channels, kH, kW)
                # Norm of each (kH, kW) filter, resulting in (out_channels, in_channels)
                diff = torch.norm(diff.flatten(2), p=2, dim=2)
                print(f"[Plotter] Info: Layer '{layer_name}' is >2D ({w1.ndim}D). Reduced to 2D for heatmap by taking L2 norm of filter kernels.")
            else:
                print(f"[Plotter] Layer '{layer_name}' has {w1.ndim} dimensions. Heatmap currently supports 1D or 2D (or 4D Conv reduced). Skipping.")
                return

        plt.figure(figsize=(10, 8))
        # Determine appropriate center for diverging colormap if values are both pos and neg
        vmin, vmax = diff.min().item(), diff.max().item()
        center = 0 if vmin < 0 < vmax else None 
        
        sns.heatmap(diff.cpu().numpy(), annot=False, cmap="coolwarm", center=center, fmt=".2f")
        s1_epoch = snapshot1.metadata.get('epoch', 'N/A')
        s2_epoch = snapshot2.metadata.get('epoch', 'N/A')
        plt.title(f'Weight Difference Heatmap: {layer_name}\n(Epoch {s2_epoch} - Epoch {s1_epoch})', fontsize=16)
        plt.xlabel('Weight Index (Dim 1)', fontsize=12)
        plt.ylabel('Weight Index (Dim 0)', fontsize=12)
        plt.tight_layout()

        if filename is None:
            safe_layer_name = layer_name.replace('.', '_').replace('[', '_').replace(']', '')
            filename = f"heatmap_diff_{safe_layer_name}_e{s1_epoch}_vs_e{s2_epoch}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"[Plotter] Plot saved: {filepath}")

# Example Usage (primary example is run_analysis_example.py)
if __name__ == '__main__':
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

    print("[INFO] Initializing Plotter with dummy comparison_results for existing plots...")
    plotter1 = Plotter(comparison_results=dummy_comparison_results, output_dir="test_plots_output/distance_plots")
    print("\n[INFO] Plotting total change over epochs (from dummy comparison_results)...")
    plotter1.plot_total_change_over_epochs()
    print("\n[INFO] Plotting per-layer change over epochs (from dummy comparison_results)...")
    plotter1.plot_per_layer_change_over_epochs()

    # Example for new trajectory and heatmap plots (needs Snapshot objects)
    # Create dummy Snapshot objects for this example
    class DummyModel(torch.nn.Module):
        def __init__(self, val1=0.0, val2=0.0, conv_val=0.0):
            super().__init__()
            self.fc1 = torch.nn.Linear(2,2, bias=True)
            self.fc1.weight.data.fill_(val1)
            self.fc1.bias.data.fill_(val1 + 0.5)
            self.output = torch.nn.Linear(2,1, bias=True)
            self.output.weight.data.fill_(val2)
            self.output.bias.data.fill_(val2 + 0.5)
            # Add a conv layer for heatmap testing with >2D tensors
            self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=2) # out, in, ks
            self.conv1.weight.data.fill_(conv_val)
            self.conv1.bias.data.fill_(conv_val + 0.1)

    snapshots_for_new_plots = [
        Snapshot(model_state=DummyModel(val1=0.1, val2=0.5, conv_val=0.01).state_dict(), metadata={"epoch": 0, "step": 0}),
        Snapshot(model_state=DummyModel(val1=0.2, val2=0.4, conv_val=0.05).state_dict(), metadata={"epoch": 1, "step": 100}),
        Snapshot(model_state=DummyModel(val1=0.3, val2=0.3, conv_val=0.1).state_dict(), metadata={"epoch": 2, "step": 200}),
        Snapshot(model_state=DummyModel(val1=0.25, val2=0.35, conv_val=0.07).state_dict(), metadata={"epoch": 3, "step": 300})
    ]

    print("\n[INFO] Initializing a new Plotter instance for trajectory/heatmap plots (with dummy snapshots)...")
    # Note: Plotter can be initialized without comparison_results if only using new methods
    plotter2 = Plotter(output_dir="test_plots_output/trajectory_heatmap_plots") 

    print("\n[INFO] Plotting specific weight trajectories...")
    plotter2.plot_specific_weight_trajectories(
        snapshots=snapshots_for_new_plots, 
        weights_to_plot={
            "fc1.weight": [(0,0), (1,1)], 
            "output.bias": [(0,)]
        }
    )

    print("\n[INFO] Plotting layer aggregate trajectories...")
    plotter2.plot_layer_aggregate_trajectories(
        snapshots=snapshots_for_new_plots,
        aggregate_fns={
            "L2_norm": lambda t: t.norm().item(),
            "mean_abs_val": lambda t: t.abs().mean().item()
        }
    )

    print("\n[INFO] Plotting layer difference heatmap (fc1.weight, epoch 0 vs 2)...")
    if len(snapshots_for_new_plots) > 2:
        plotter2.plot_layer_difference_heatmap(
            snapshots_for_new_plots[0], 
            snapshots_for_new_plots[2], 
            layer_name="fc1.weight"
        )
        print("\n[INFO] Plotting layer difference heatmap (conv1.weight, epoch 0 vs 2)...")
        plotter2.plot_layer_difference_heatmap(
            snapshots_for_new_plots[0], 
            snapshots_for_new_plots[2], 
            layer_name="conv1.weight"
        )
        print("\n[INFO] Plotting layer difference heatmap for a bias (output.bias, epoch 0 vs 1)...")
        plotter2.plot_layer_difference_heatmap(
            snapshots_for_new_plots[0], 
            snapshots_for_new_plots[1], 
            layer_name="output.bias"
        )

    print("\n[INFO] Plotting example finished. Check the 'test_plots_output' directory for subfolders.") 