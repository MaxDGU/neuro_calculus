import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import numpy as np # For histogram binning if needed
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

    def _sanitize_filename_component(self, component: str) -> str:
        """Sanitizes a string component for use in a filename."""
        # Replace problematic characters with underscores
        return str(component).replace('.', '_').replace('[', '_').replace(']', '').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')

    def _get_x_axis_values_and_label(self, items_with_metadata: List[Any]) -> Tuple[List[Any], str]:
        """
        Determines x-axis values and a label from a list of items having metadata.
        Items can be Snapshots or metadata dictionaries.
        Prioritizes 'epoch', then 'step', then list index.
        """
        x_values = []
        labels_found = set()

        for i, item in enumerate(items_with_metadata):
            meta = {}
            if isinstance(item, Snapshot):
                meta = item.metadata
            elif isinstance(item, dict): # Assuming it's a metadata dict itself or part of comparison result
                if "snapshot1_metadata" in item and "snapshot2_metadata" in item: # Comparison dict
                    meta = item.get("snapshot2_metadata", {}) # Use metadata of the second snapshot for progression
                else: # Direct metadata dict
                    meta = item
            
            if meta.get("epoch") is not None:
                x_values.append(meta["epoch"])
                labels_found.add("Epoch")
            elif meta.get("step") is not None:
                x_values.append(meta["step"])
                labels_found.add("Step")
            else:
                x_values.append(i)
                labels_found.add("Snapshot Index")
        
        if "Epoch" in labels_found:
            x_label = "Epoch"
        elif "Step" in labels_found:
            x_label = "Step"
        else:
            x_label = "Snapshot Index"
        
        # If mixed, and epoch is present, prefer epoch for consistency if some items only have index
        # This logic ensures if ANY item has epoch, label is epoch. If ANY has step (and no epoch), label is step.
        if not x_values: # Should not happen if items_with_metadata is not empty
            return [], "Snapshot Index"
            
        return x_values, x_label

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

        # Use metadata from the second snapshot in each pair for the x-axis
        # Each item in self.comparison_results is a dict that should contain snapshot2_metadata
        x_values, x_label = self._get_x_axis_values_and_label(self.comparison_results)
        total_changes = []

        for i, comp_res in enumerate(self.comparison_results):
            change_val = comp_res.get("aggregate_L2_distance_from_layers")
            if change_val is None or change_val < 0: 
                print(f"[Plotter] Warning: Missing/invalid aggregate L2 change for item {i}. Using 0.")
                total_changes.append(0)
            else:
                total_changes.append(change_val)
        
        if not x_values or not total_changes or len(x_values) != len(total_changes):
            print("[Plotter] Insufficient data for plotting total change after processing.")
            return

        plt.figure(figsize=(12, 7))
        plt.plot(x_values, total_changes, marker='o', linestyle='-', color='b')
        plt.title('Total Model Weight Change (Aggregate L2 Distance) Over Training Progression', fontsize=16)
        plt.xlabel(f"{x_label} of Second Snapshot in Pair", fontsize=12)
        plt.ylabel('Aggregate L2 Distance', fontsize=12)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, self._sanitize_filename_component(filename))
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

        layer_data: Dict[str, Dict[str, List[Any]]] = {}
        overall_x_values, overall_x_label = self._get_x_axis_values_and_label(self.comparison_results)

        for i, comp_res in enumerate(self.comparison_results):
            if "per_layer_L2_distance" in comp_res and isinstance(comp_res["per_layer_L2_distance"], dict):
                for layer_name, dist in comp_res["per_layer_L2_distance"].items():
                    if layer_name not in layer_data:
                        layer_data[layer_name] = {"x_vals": [None] * len(overall_x_values), "changes": [None] * len(overall_x_values)}
                    
                    layer_data[layer_name]["x_vals"][i] = overall_x_values[i]
                    if dist is None or dist < 0:
                        print(f"[Plotter] Warning: Missing/invalid L2 for layer '{layer_name}', item {i}. Using 0.")
                        layer_data[layer_name]["changes"][i] = 0
                    else:
                        layer_data[layer_name]["changes"][i] = dist
            else:
                 print(f"[Plotter] Warning: Missing 'per_layer_L2_distance' in comparison item {i}.")

        if not layer_data:
            print("[Plotter] No per-layer L2 data found to plot.")
            return

        for layer_name in sorted(layer_data.keys()):
            # Filter out None placeholders if some comparisons didn't have this layer (shouldn't happen with current Comparer)
            current_x_vals = [x for x,y in zip(layer_data[layer_name]["x_vals"], layer_data[layer_name]["changes"]) if y is not None]
            current_changes = [y for y in layer_data[layer_name]["changes"] if y is not None]
            
            if not current_x_vals or not current_changes:
                print(f"[Plotter] No valid data points to plot for layer: {layer_name}")
                continue

            plt.figure(figsize=(12, 7))
            plt.plot(current_x_vals, current_changes, marker='o', linestyle='-')
            plt.title(f'L2 Distance Change for Layer: {layer_name}', fontsize=16)
            plt.xlabel(f"{overall_x_label} of Second Snapshot in Pair", fontsize=12)
            plt.ylabel('L2 Distance', fontsize=12)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            
            safe_layer_name = self._sanitize_filename_component(layer_name)
            plot_filename = f"{self._sanitize_filename_component(filename_prefix)}_{safe_layer_name}.png"
            filepath = os.path.join(self.output_dir, plot_filename)
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

        overall_x_values, overall_x_label = self._get_x_axis_values_and_label(snapshots)
        plot_data_collected: Dict[str, Dict[str, List[Any]]] = {}

        for i, snap in enumerate(snapshots):
            for layer_name, indices_list in weights_to_plot.items():
                if layer_name not in snap.model_state:
                    continue
                layer_tensor = snap.model_state[layer_name]
                for index_tuple in indices_list:
                    try:
                        weight_val = layer_tensor[index_tuple].item()
                        index_str_sanitized = self._sanitize_filename_component(str(index_tuple))
                        dict_key = f"{self._sanitize_filename_component(layer_name)}_idx_{index_str_sanitized}"

                        if dict_key not in plot_data_collected:
                            plot_data_collected[dict_key] = {"x_vals": [], "values": [], "layer": layer_name, "index_str": str(index_tuple)}
                        
                        plot_data_collected[dict_key]["x_vals"].append(overall_x_values[i])
                        plot_data_collected[dict_key]["values"].append(weight_val)
                    except IndexError:
                        print(f"[Plotter] Index {index_tuple} out of bounds for '{layer_name}' in snapshot {i}.")
                    except Exception as e:
                        print(f"[Plotter] Error accessing {layer_name}{index_tuple} in snapshot {i}: {e}")

        if not plot_data_collected:
            print("[Plotter] No data collected for specific weight trajectories.")
            return

        for key, data in plot_data_collected.items():
            if not data["x_vals"] or not data["values"]:
                continue
            
            plt.figure(figsize=(12, 7))
            plt.plot(data["x_vals"], data["values"], marker='o', linestyle='-')
            plt.title(f'Trajectory for Weight: {data["layer"]}{data["index_str"]}', fontsize=16)
            plt.xlabel(overall_x_label, fontsize=12)
            plt.ylabel('Weight Value', fontsize=12)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            
            plot_filename = f"{self._sanitize_filename_component(filename_prefix)}_{key}.png"
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

        overall_x_values, overall_x_label = self._get_x_axis_values_and_label(snapshots)
        # Data: { agg_fn_name: { layer_name: {"x_vals": [], "values": []} } }
        plot_data_collected: Dict[str, Dict[str, Dict[str, List[Any]]]] = {fn_name: {} for fn_name in aggregate_fns.keys()}
        
        for i, snap in enumerate(snapshots):
            for layer_name, layer_tensor in snap.model_state.items():
                if layer_filter and layer_name not in layer_filter:
                    continue
                for fn_name, agg_fn in aggregate_fns.items():
                    try:
                        stat_val = agg_fn(layer_tensor)
                        if layer_name not in plot_data_collected[fn_name]:
                            plot_data_collected[fn_name][layer_name] = {"x_vals": [], "values": []}
                        plot_data_collected[fn_name][layer_name]["x_vals"].append(overall_x_values[i])
                        plot_data_collected[fn_name][layer_name]["values"].append(stat_val)
                    except Exception as e:
                        print(f"[Plotter] Error computing aggregate '{fn_name}' for '{layer_name}' in snapshot {i}: {e}")

        if not any(plot_data_collected.values()):
            print("[Plotter] No data collected for layer aggregate trajectories.")
            return

        for fn_name, layer_data_map in plot_data_collected.items():
            if not layer_data_map: continue
            plt.figure(figsize=(12, 7))
            for layer_name, data in sorted(layer_data_map.items()):
                if data["x_vals"] and data["values"]:
                    plt.plot(data["x_vals"], data["values"], marker='o', linestyle='-', label=layer_name)
            
            plt.title(f'Trajectory of Layer Aggregate: {fn_name}', fontsize=16)
            plt.xlabel(overall_x_label, fontsize=12)
            plt.ylabel('Aggregate Value', fontsize=12)
            if any(data["x_vals"] for data in layer_data_map.values()): # Only add legend if there's data
                plt.legend(loc='best', fontsize=10)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            
            plot_filename = f"{self._sanitize_filename_component(filename_prefix)}_{self._sanitize_filename_component(fn_name)}.png"
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
            print(f"[Plotter] Layer '{layer_name}' not found in one or both snapshots for heatmap.")
            return
        w1, w2 = snapshot1.model_state[layer_name], snapshot2.model_state[layer_name]
        if w1.shape != w2.shape:
            print(f"[Plotter] Shape mismatch for '{layer_name}': {w1.shape} vs {w2.shape} for heatmap.")
            return

        diff_tensor = w2 - w1
        plot_values = torch.zeros_like(diff_tensor) # Placeholder
        mode_title_prefix = ""

        if mode == "raw":
            plot_values = diff_tensor
            mode_title_prefix = "Raw"
        elif mode == "percentage":
            plot_values = diff_tensor / (torch.abs(w1) + self.epsilon)
            mode_title_prefix = "Percentage"
        elif mode == "standardized":
            mean_diff = torch.mean(diff_tensor.float())
            std_diff = torch.std(diff_tensor.float())
            plot_values = (diff_tensor.float() - mean_diff) / (std_diff + self.epsilon)
            mode_title_prefix = "Standardized (Z-score)"
        else:
            print(f"[Plotter] Unknown heatmap mode '{mode}'. Defaulting to 'raw'.")
            plot_values = diff_tensor
            mode_title_prefix = "Raw"
        
        original_ndim = plot_values.ndim
        y_label, x_label = "Weight Index (Dim 0)", "Weight Index (Dim 1)"
        if original_ndim == 1: 
            plot_values = plot_values.unsqueeze(0)
            y_label, x_label = f"{layer_name} (Bias)", "Bias Element Index"
        elif original_ndim == 0:
             print(f"[Plotter] Layer '{layer_name}' is scalar. Heatmap not suitable.")
             return
        elif original_ndim == 4:
            plot_values = torch.norm(plot_values.flatten(2), p=2, dim=2)
            print(f"[Plotter] Info: 4D Layer '{layer_name}' heatmap shows L2 norm of '{mode}' change per filter.")
            y_label, x_label = "Output Channel Index (Dim 0)", "Input Channel Index (Dim 1)"
        elif original_ndim > 2 and original_ndim != 4: # Other >2D tensors
            print(f"[Plotter] Warning: Layer '{layer_name}' has {original_ndim} dims. Flattening for heatmap.")
            try:
                plot_values = plot_values.reshape(-1, plot_values.shape[-1])
                y_label, x_label = "Flattened Dims", f"Original Dim {original_ndim-1} Index"
            except Exception as e:
                print(f"[Plotter] Failed to reshape {original_ndim}D tensor for '{layer_name}': {e}. Skipping heatmap.")
                return
        
        plt.figure(figsize=(10, 8))
        center_val = 0 if mode in ["raw", "standardized"] or (plot_values.min().item() < 0 and plot_values.max().item() > 0) else None
        sns.heatmap(plot_values.cpu().numpy(), annot=False, cmap="coolwarm", center=center_val, robust=robust_colormap, fmt=".2f")
        s1_id = f"S{snapshot1.metadata.get('epoch','?')}(st{snapshot1.metadata.get('step','?')})"
        s2_id = f"S{snapshot2.metadata.get('epoch','?')}(st{snapshot2.metadata.get('step','?')})"
        plt.title(f'{mode_title_prefix} Change Heatmap: {layer_name}\n({s2_id} - {s1_id})', fontsize=14)
        plt.xlabel(x_label, fontsize=12); plt.ylabel(y_label, fontsize=12)
        plt.tight_layout()

        if filename is None:
            fn = f"heatmap_{mode}_{self._sanitize_filename_component(layer_name)}_{s1_id}_vs_{s2_id}.png"
        else:
            fn = self._sanitize_filename_component(filename)
        filepath = os.path.join(self.output_dir, fn)
        plt.savefig(filepath); plt.close()
        print(f"[Plotter] Plot saved: {filepath}")

    def plot_layer_change_distribution(self, 
                                       snapshot1: Snapshot, 
                                       snapshot2: Snapshot, 
                                       layer_name: str,
                                       mode: str = "raw", 
                                       bins: int = 50,
                                       filename: Optional[str] = None):
        """
        Plots a histogram of the distribution of weight changes in a specific layer.

        Args:
            snapshot1 (Snapshot): The first snapshot.
            snapshot2 (Snapshot): The second snapshot.
            layer_name (str): The name of the layer to analyze.
            mode (str, optional): Mode for calculating differences ('raw', 'percentage', 'standardized').
                                Defaults to "raw".
            bins (int, optional): Number of bins for the histogram. Defaults to 50.
            filename (Optional[str], optional): Name for the output plot file. Defaults to None.
        """
        if layer_name not in snapshot1.model_state or layer_name not in snapshot2.model_state:
            print(f"[Plotter] Layer '{layer_name}' not in one or both snapshots for distribution plot.")
            return
        w1, w2 = snapshot1.model_state[layer_name], snapshot2.model_state[layer_name]
        if w1.shape != w2.shape:
            print(f"[Plotter] Shape mismatch for '{layer_name}': {w1.shape} vs {w2.shape} for distribution.")
            return

        diff_tensor = w2 - w1
        plot_values = torch.zeros_like(diff_tensor) # Placeholder
        mode_title_prefix = ""
        x_hist_label = "Change Value"

        if mode == "raw":
            plot_values = diff_tensor
            mode_title_prefix = "Raw"
            x_hist_label = "Raw Difference (W2 - W1)"
        elif mode == "percentage":
            plot_values = diff_tensor / (torch.abs(w1) + self.epsilon)
            mode_title_prefix = "Percentage"
            x_hist_label = "Percentage Change ((W2 - W1) / |W1|)"
        elif mode == "standardized":
            mean_diff = torch.mean(diff_tensor.float())
            std_diff = torch.std(diff_tensor.float())
            plot_values = (diff_tensor.float() - mean_diff) / (std_diff + self.epsilon)
            mode_title_prefix = "Standardized (Z-score)"
            x_hist_label = "Standardized Difference (Z-score)"
        else:
            print(f"[Plotter] Unknown distribution mode '{mode}'. Defaulting to 'raw'.")
            plot_values = diff_tensor
            mode_title_prefix = "Raw"
            x_hist_label = "Raw Difference (W2 - W1)"
        
        flattened_values = plot_values.flatten().cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(flattened_values, bins=bins, edgecolor='black', alpha=0.7)
        s1_id = f"S{snapshot1.metadata.get('epoch','?')}(st{snapshot1.metadata.get('step','?')})"
        s2_id = f"S{snapshot2.metadata.get('epoch','?')}(st{snapshot2.metadata.get('step','?')})"
        plt.title(f'{mode_title_prefix} Change Distribution: {layer_name}\n({s2_id} - {s1_id})', fontsize=14)
        plt.xlabel(x_hist_label, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()

        if filename is None:
            fn = f"hist_dist_{mode}_{self._sanitize_filename_component(layer_name)}_{s1_id}_vs_{s2_id}.png"
        else:
            fn = self._sanitize_filename_component(filename)
        filepath = os.path.join(self.output_dir, fn)
        plt.savefig(filepath); plt.close()
        print(f"[Plotter] Plot saved: {filepath}")

# Example Usage (primary example is run_analysis_example.py)
if __name__ == '__main__':
    # Dummy model and snapshots for testing
    class DummyModel(torch.nn.Module):
        def __init__(self, val_offset=0.0):
            super().__init__()
            self.fc1_w = nn.Parameter(torch.randn(10,5) * 0.1 + val_offset)
            self.fc1_b = nn.Parameter(torch.randn(10) * 0.1 + val_offset)
            self.conv1_w = nn.Parameter(torch.randn(4,2,3,3) * 0.1 + val_offset)

        def state_dict(self, destination=None, prefix='', keep_vars=False):
            return {'fc1.weight': self.fc1_w, 'fc1.bias': self.fc1_b, 'conv1.weight': self.conv1_w}

    s_initial = Snapshot(model_state=DummyModel(val_offset=0.0).state_dict(), metadata={"epoch": 0, "step": 0})
    s_epoch5 = Snapshot(model_state=DummyModel(val_offset=0.05).state_dict(), metadata={"epoch": 5, "step": 500})
    s_epoch10 = Snapshot(model_state=DummyModel(val_offset=-0.02).state_dict(), metadata={"epoch": 10, "step": 1000})
    all_snaps = [s_initial, s_epoch5, s_epoch10]

    # Dummy comparison results for L2 plots
    comp_res = [
        {"snapshot1_metadata": s_initial.metadata, "snapshot2_metadata": s_epoch5.metadata, 
         "aggregate_L2_distance_from_layers": 0.5, 
         "per_layer_L2_distance": {"fc1.weight": 0.3, "fc1.bias": 0.1, "conv1.weight":0.2}},
        {"snapshot1_metadata": s_epoch5.metadata, "snapshot2_metadata": s_epoch10.metadata, 
         "aggregate_L2_distance_from_layers": 0.8, 
         "per_layer_L2_distance": {"fc1.weight": 0.4, "fc1.bias": 0.15, "conv1.weight":0.35}}
    ]

    test_plot_dir = "test_plotter_output_refined"
    if os.path.exists(test_plot_dir): shutil.rmtree(test_plot_dir)
    plotter = Plotter(comparison_results=comp_res, output_dir=test_plot_dir)

    print("--- Testing L2 Distance Plots (with x-axis helper) ---")
    plotter.plot_total_change_over_epochs()
    plotter.plot_per_layer_change_over_epochs()

    print("--- Testing Trajectory Plots (with x-axis helper) ---")
    plotter.plot_specific_weight_trajectories(all_snaps, {"fc1.weight":[(0,0),(1,1)], "conv1.weight":[(0,0,0,0)]})
    plotter.plot_layer_aggregate_trajectories(all_snaps, {"L2_Norm": lambda t: t.norm().item()})
    
    print("--- Testing Refined Heatmaps & New Distribution Plots ---")
    layers_to_test = ["fc1.weight", "fc1.bias", "conv1.weight"]
    modes = ["raw", "percentage", "standardized"]

    for layer in layers_to_test:
        for mode in modes:
            print(f"Plotting Heatmap: {layer}, mode: {mode}")
            plotter.plot_layer_difference_heatmap(s_initial, s_epoch10, layer, mode=mode)
            print(f"Plotting Distribution: {layer}, mode: {mode}")
            plotter.plot_layer_change_distribution(s_initial, s_epoch10, layer, mode=mode, bins=30)

    print(f"\nAll test plots saved in {test_plot_dir}") 