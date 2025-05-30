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
    2. A list of `Snapshot` objects directly for plots like weight trajectories or heatmaps.

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
        self.epsilon = 1e-8 # For stable division in percentage change

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
        x_axis_label = "Epoch / Snapshot Index of Second Snapshot in Pair"

        for i, comp in enumerate(self.comparison_results):
            epoch_val = None
            metadata2 = comp.get("snapshot2_metadata", {})
            
            if metadata2.get("epoch") is not None:
                epoch_val = metadata2["epoch"]
                x_axis_label = "Epoch of Second Snapshot in Pair"
            elif metadata2.get("step") is not None: # Fallback to step
                epoch_val = metadata2["step"]
                x_axis_label = "Step of Second Snapshot in Pair"
            else: # Fallback to index
                epoch_val = i 
            epochs.append(epoch_val)
            
            change_val = comp.get("aggregate_L2_distance_from_layers")
            if change_val is None or change_val < 0: 
                print(f"[Plotter] Warning: Missing or invalid aggregate change for comparison index {i} (data: {metadata2}). Using 0 for plot.")
                total_changes.append(0) 
            else:
                total_changes.append(change_val)

        if not epochs:
            print("[Plotter] Could not extract sufficient data for plotting total change.")
            return

        plt.figure(figsize=(12, 7))
        plt.plot(epochs, total_changes, marker='o', linestyle='-', color='b')
        plt.title('Total Model Weight Change (Aggregate L2 Distance) Over Training Progression', fontsize=16)
        plt.xlabel(x_axis_label, fontsize=12)
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
        x_axis_label_base = "Epoch / Snapshot Index of Second Snapshot in Pair"


        for i, comp in enumerate(self.comparison_results):
            x_val = None
            metadata2 = comp.get("snapshot2_metadata", {})
            current_x_axis_label = x_axis_label_base # Default for this comparison

            if metadata2.get("epoch") is not None:
                x_val = metadata2["epoch"]
                current_x_axis_label = "Epoch of Second Snapshot in Pair"
            elif metadata2.get("step") is not None:
                x_val = metadata2["step"]
                current_x_axis_label = "Step of Second Snapshot in Pair"
            else:
                x_val = i 

            if "per_layer_L2_distance" in comp and isinstance(comp["per_layer_L2_distance"], dict):
                for layer_name, dist in comp["per_layer_L2_distance"].items():
                    if layer_name in plot_data: 
                        plot_data[layer_name]["x_axis_vals"].append(x_val)
                        if dist is None or dist < 0: 
                            print(f"[Plotter] Warning: Missing or invalid L2 distance for layer '{layer_name}' at index {i} (data: {metadata2}). Using 0.")
                            plot_data[layer_name]["changes"].append(0)
                        else:
                            plot_data[layer_name]["changes"].append(dist)
                        # Store the most specific x-axis label found
                        if "x_label" not in plot_data[layer_name] or current_x_axis_label != x_axis_label_base :
                             plot_data[layer_name]["x_label"] = current_x_axis_label
        
        if not any(plot_data[name]["x_axis_vals"] for name in layer_names):
            print("[Plotter] No valid data points found to plot for any layer.")
            return

        for layer_name in sorted(list(layer_names)):
            data = plot_data[layer_name]
            if not data["x_axis_vals"] or not data["changes"]:
                print(f"[Plotter] No data to plot for layer: {layer_name}")
                continue
            
            final_x_label = data.get("x_label", x_axis_label_base)

            plt.figure(figsize=(12, 7))
            plt.plot(data["x_axis_vals"], data["changes"], marker='o', linestyle='-')
            plt.title(f'Weight Change (L2 Distance) for Layer: {layer_name}', fontsize=16)
            plt.xlabel(final_x_label, fontsize=12)
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

        plot_data: Dict[str, Dict[str, List[Any]]] = {}
        x_axis_label = "Epoch / Snapshot Index" # Default

        for i, snap in enumerate(snapshots):
            current_x_val = i # Default to index
            current_x_axis_label = "Snapshot Index"

            if "epoch" in snap.metadata and snap.metadata["epoch"] is not None:
                current_x_val = snap.metadata["epoch"]
                current_x_axis_label = "Epoch"
            elif "step" in snap.metadata and snap.metadata["step"] is not None: # Fallback to step
                current_x_val = snap.metadata["step"]
                current_x_axis_label = "Step"
            
            # Use the most specific label found across snapshots for consistency in plot titles
            if current_x_axis_label != "Snapshot Index" and x_axis_label == "Epoch / Snapshot Index":
                x_axis_label = current_x_axis_label
            elif current_x_axis_label == "Epoch" and x_axis_label == "Step": # Prioritize Epoch if both step/epoch seen
                x_axis_label = "Epoch"


            for layer_name, indices_list in weights_to_plot.items():
                if layer_name not in snap.model_state:
                    continue
                
                layer_tensor = snap.model_state[layer_name]
                for index_tuple in indices_list:
                    try:
                        weight_val = layer_tensor[index_tuple].item()
                        # Sanitize index_tuple for filename and dict key
                        index_str_sanitized = str(index_tuple).replace(', ', '_').replace('(', '').replace(')', '').replace(',', '')
                        dict_key = f"{layer_name}_idx_{index_str_sanitized}"


                        if dict_key not in plot_data:
                            plot_data[dict_key] = {"x_axis_vals": [], "values": [], "layer": layer_name, "index": str(index_tuple), "x_label_type": x_axis_label}
                        
                        plot_data[dict_key]["x_axis_vals"].append(current_x_val)
                        plot_data[dict_key]["values"].append(weight_val)
                        # Update general x-axis label if a more specific one (epoch/step) is found
                        if x_axis_label == "Epoch / Snapshot Index" and current_x_axis_label != "Snapshot Index":
                             plot_data[dict_key]["x_label_type"] = current_x_axis_label
                        elif x_axis_label == "Step" and current_x_axis_label == "Epoch":
                             plot_data[dict_key]["x_label_type"] = "Epoch"


                    except IndexError:
                        print(f"[Plotter] Warning: Index {index_tuple} out of bounds for layer '{layer_name}' in snapshot {i} (metadata: {snap.metadata}).")
                    except Exception as e:
                        print(f"[Plotter] Error accessing weight {layer_name}{index_tuple} in snapshot {i} (metadata: {snap.metadata}): {e}")

        if not plot_data:
            print("[Plotter] No valid data collected for specific weight trajectories.")
            return

        final_x_label_type = "Snapshot Index" # Fallback
        # Determine the most specific overall x-axis label to use from all collected data
        labels_seen = {data.get("x_label_type", "Snapshot Index") for data in plot_data.values()}
        if "Epoch" in labels_seen: final_x_label_type = "Epoch"
        elif "Step" in labels_seen: final_x_label_type = "Step"


        for key, data in plot_data.items():
            if not data["x_axis_vals"] or not data["values"]:
                continue
            
            plt.figure(figsize=(12, 7))
            plt.plot(data["x_axis_vals"], data["values"], marker='o', linestyle='-')
            plt.title(f'Trajectory for Weight: {data["layer"]}{data["index"]}', fontsize=16)
            plt.xlabel(final_x_label_type, fontsize=12)
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

        plot_data: Dict[str, Dict[str, Dict[str, List[Any]]]] = {fn_name: {} for fn_name in aggregate_fns.keys()}
        processed_layers = set()
        determined_x_label = "Snapshot Index" # Fallback

        for i, snap in enumerate(snapshots):
            current_x_val = i
            current_x_label_type = "Snapshot Index"

            if "epoch" in snap.metadata and snap.metadata["epoch"] is not None:
                current_x_val = snap.metadata["epoch"]
                current_x_label_type = "Epoch"
            elif "step" in snap.metadata and snap.metadata["step"] is not None:
                current_x_val = snap.metadata["step"]
                current_x_label_type = "Step"

            if determined_x_label == "Snapshot Index" and current_x_label_type != "Snapshot Index":
                determined_x_label = current_x_label_type
            elif determined_x_label == "Step" and current_x_label_type == "Epoch":
                determined_x_label = "Epoch" # Prioritize Epoch

            for layer_name, layer_tensor in snap.model_state.items():
                if layer_filter and layer_name not in layer_filter:
                    continue
                processed_layers.add(layer_name)

                for fn_name, agg_fn in aggregate_fns.items():
                    try:
                        stat_val = agg_fn(layer_tensor)
                        if layer_name not in plot_data[fn_name]:
                            plot_data[fn_name][layer_name] = {"x_axis_vals": [], "values": []}
                        
                        plot_data[fn_name][layer_name]["x_axis_vals"].append(current_x_val)
                        plot_data[fn_name][layer_name]["values"].append(stat_val)
                    except Exception as e:
                        print(f"[Plotter] Error computing aggregate '{fn_name}' for layer '{layer_name}' in snap {i} (meta: {snap.metadata}): {e}")

        if not processed_layers:
            print("[Plotter] No layers processed for aggregate trajectories (check filter or snapshot content).")
            return

        for fn_name, layer_data_map in plot_data.items():
            plt.figure(figsize=(12, 7))
            for layer_name, data in sorted(layer_data_map.items()):
                if data["x_axis_vals"] and data["values"]:
                    plt.plot(data["x_axis_vals"], data["values"], marker='o', linestyle='-', label=layer_name)
            
            plt.title(f'Trajectory of Layer Aggregate: {fn_name}', fontsize=16)
            plt.xlabel(determined_x_label, fontsize=12) # Use the determined label
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
                                      mode: str = "raw", # "raw", "percentage", "standardized"
                                      robust_colormap: bool = True,
                                      filename: Optional[str] = None):
        """
        Plots a heatmap of the difference between a specific layer's weights 
        from two snapshots, with different modes of calculating the difference.

        For 4D convolutional layers (e.g., [out_ch, in_ch, kH, kW]), the difference
        tensor (W2 - W1) is first computed. Then, for each filter (i,j), the L2 norm 
        of its [kH, kW] difference components is taken. The heatmap then visualizes 
        this 2D grid ([out_ch, in_ch]) of filter change magnitudes.

        Args:
            snapshot1 (Snapshot): The first snapshot.
            snapshot2 (Snapshot): The second snapshot.
            layer_name (str): The name of the layer to compare (e.g., 'fc1.weight').
            mode (str, optional): The mode for calculating differences.
                - "raw": W2 - W1 (default)
                - "percentage": (W2 - W1) / (abs(W1) + epsilon)
                - "standardized": (Diff - Diff.mean()) / (Diff.std() + epsilon), where Diff = W2 - W1
            robust_colormap (bool, optional): If True, uses robust quantiles for 
                                              colormap scaling (sns.heatmap robust=True).
                                              Defaults to True.
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
            print(f"[Plotter] Shapes of layer '{layer_name}' differ: {w1.shape} vs {w2.shape}. Cannot plot heatmap.")
            return
        
        diff_tensor = w2 - w1
        mode_title_prefix = ""

        if mode == "raw":
            plot_values = diff_tensor
            mode_title_prefix = "Raw"
        elif mode == "percentage":
            plot_values = diff_tensor / (torch.abs(w1) + self.epsilon)
            mode_title_prefix = "Percentage"
        elif mode == "standardized":
            mean_diff = torch.mean(diff_tensor.float()) # Ensure float for mean/std
            std_diff = torch.std(diff_tensor.float())
            plot_values = (diff_tensor - mean_diff) / (std_diff + self.epsilon)
            mode_title_prefix = "Standardized (Z-score)"
        else:
            print(f"[Plotter] Unknown heatmap mode: {mode}. Defaulting to 'raw'.")
            plot_values = diff_tensor
            mode_title_prefix = "Raw"
        
        original_ndim = plot_values.ndim
        if original_ndim == 1: # For bias vectors, reshape to 2D for heatmap
            plot_values = plot_values.unsqueeze(0)
            y_label = f"{layer_name} (Bias)"
            x_label = "Bias Element Index"
        elif original_ndim == 0: # Scalar tensor
             print(f"[Plotter] Layer '{layer_name}' is a scalar. Heatmap not suitable.")
             return
        elif original_ndim > 2:
            if original_ndim == 4: # Typical Conv2D: (out_channels, in_channels, kH, kW)
                # Reduce to 2D by taking L2 norm of the change components within each filter
                plot_values = torch.norm(plot_values.flatten(2), p=2, dim=2)
                print(f"[Plotter] Info: Layer '{layer_name}' is 4D. Heatmap shows L2 norm of '{mode}' change for each [out_ch, in_ch] filter.")
                y_label = "Output Channel Index (Dim 0)"
                x_label = "Input Channel Index (Dim 1)"
            else: # Other >2D tensors
                print(f"[Plotter] Layer '{layer_name}' has {original_ndim} dims. Heatmap for >2D (non-4D) not implemented. Trying to flatten last two dims if possible.")
                try:
                    h, w = plot_values.shape[-2], plot_values.shape[-1]
                    plot_values = plot_values.reshape(-1, w) # Flatten all but last dim
                    y_label = "Flattened Dims"
                    x_label = f"Original Dim {original_ndim-1} Index"
                except:
                    print(f"[Plotter] Could not reshape {original_ndim}D tensor for heatmap. Skipping.")
                    return
        else: # 2D tensor (e.g. Linear layer weights)
            y_label = "Output Unit Index (Dim 0)"
            x_label = "Input Unit Index (Dim 1)"


        plt.figure(figsize=(10, 8))
        center_val = 0 if mode in ["raw", "standardized"] or (plot_values.min() < 0 and plot_values.max() > 0) else None
        
        sns.heatmap(plot_values.cpu().numpy(), annot=False, cmap="coolwarm", center=center_val, robust=robust_colormap, fmt=".2f")
        
        s1_epoch = snapshot1.metadata.get('epoch', 'S1')
        s2_epoch = snapshot2.metadata.get('epoch', 'S2')
        s1_step = snapshot1.metadata.get('step', 's1')
        s2_step = snapshot2.metadata.get('step', 's2')

        title = (f'{mode_title_prefix} Change Heatmap: {layer_name}\n'
                 f'Snap {s2_epoch}(st{s2_step}) - Snap {s1_epoch}(st{s1_step})')
        plt.title(title, fontsize=14)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.tight_layout()

        if filename is None:
            safe_layer_name = layer_name.replace('.', '_').replace('[', '_').replace(']', '')
            filename = f"heatmap_{mode}_{safe_layer_name}_e{s1_epoch}_vs_e{s2_epoch}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"[Plotter] Plot saved: {filepath}")

# Example Usage (primary example is run_analysis_example.py)
if __name__ == '__main__':
    dummy_comparison_results = [
        {"snapshot1_metadata": {"epoch": 0, "step": 0}, "snapshot2_metadata": {"epoch": 1, "step": 100}, "aggregate_L2_distance_from_layers": 0.5, "per_layer_L2_distance": {"fc1.weight": 0.3, "fc2.bias": 0.1}},
        {"snapshot1_metadata": {"epoch": 1, "step": 100}, "snapshot2_metadata": {"epoch": 2, "step": 200}, "aggregate_L2_distance_from_layers": 0.8, "per_layer_L2_distance": {"fc1.weight": 0.4, "fc2.bias": 0.15}},
    ]

    print("[INFO] Initializing Plotter with dummy comparison_results for L2 distance plots...")
    plotter1 = Plotter(comparison_results=dummy_comparison_results, output_dir="test_plots_output/distance_plots")
    plotter1.plot_total_change_over_epochs()
    plotter1.plot_per_layer_change_over_epochs()

    class DummyModel(torch.nn.Module):
        def __init__(self, val_offset=0.0):
            super().__init__()
            self.fc1_w = nn.Parameter(torch.tensor([[0.1, 0.2], [0.3, 0.4]]) + val_offset)
            self.fc1_b = nn.Parameter(torch.tensor([0.05, 0.15]) + val_offset)
            self.conv1_w = nn.Parameter(torch.randn(2,1,2,2) + val_offset) # out,in,kH,kW
            self.scalar_param = nn.Parameter(torch.tensor(1.0) + val_offset)

        def state_dict(self, destination=None, prefix='', keep_vars=False):
            # Simplified state_dict for example
            return {
                'fc1.weight': self.fc1_w, 
                'fc1.bias': self.fc1_b,
                'conv1.weight': self.conv1_w,
                'scalar_param': self.scalar_param
            }

    snapshots_for_heatmaps = [
        Snapshot(model_state=DummyModel(val_offset=0.0).state_dict(), metadata={"epoch": 0, "step": 0}),
        Snapshot(model_state=DummyModel(val_offset=0.1).state_dict(), metadata={"epoch": 1, "step": 100}),
        Snapshot(model_state=DummyModel(val_offset=-0.05).state_dict(), metadata={"epoch": 2, "step": 200})
    ]

    print("\n[INFO] Initializing a new Plotter for trajectory/heatmap plots...")
    plotter2 = Plotter(output_dir="test_plots_output/trajectory_heatmap_plots") 

    print("\n[INFO] Plotting layer difference heatmaps with different modes...")
    s1 = snapshots_for_heatmaps[0]
    s2 = snapshots_for_heatmaps[1]
    s3 = snapshots_for_heatmaps[2]

    layers_to_heatmap = ["fc1.weight", "fc1.bias", "conv1.weight", "scalar_param"]
    modes_to_test = ["raw", "percentage", "standardized"]

    for layer in layers_to_heatmap:
        for mode in modes_to_test:
            print(f"  Testing heatmap: Layer '{layer}', Mode '{mode}' (Snap 0 vs 1)")
            plotter2.plot_layer_difference_heatmap(s1, s2, layer_name=layer, mode=mode, robust_colormap=True)
            if layer == "fc1.weight": # Test non-robust and specific filename
                 print(f"  Testing heatmap: Layer '{layer}', Mode '{mode}' (Snap 0 vs 2, non-robust)")
                 plotter2.plot_layer_difference_heatmap(s1, s3, layer_name=layer, mode=mode, robust_colormap=False, filename=f"custom_heatmap_{layer}_{mode}_s1s3.png")


    print("\n[INFO] Plotting specific weight trajectories...")
    plotter2.plot_specific_weight_trajectories(
        snapshots=snapshots_for_heatmaps, 
        weights_to_plot={"fc1.weight": [(0,0), (1,1)], "fc1.bias": [(0,)]}
    )

    print("\n[INFO] Plotting layer aggregate trajectories...")
    plotter2.plot_layer_aggregate_trajectories(
        snapshots=snapshots_for_heatmaps,
        aggregate_fns={"L2_norm": lambda t: t.norm().item(), "mean_abs": lambda t: t.abs().mean().item()}
    )
    print("\n[INFO] Plotting example finished. Check 'test_plots_output' subdirectories.") 