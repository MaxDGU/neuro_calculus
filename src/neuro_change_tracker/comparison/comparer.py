import torch
from typing import Dict, List, Any # Removed Tuple as it wasn't used
from ..core.snapshot import Snapshot

class Comparer:
    """
    Provides methods to compare model snapshots, primarily by calculating L2 distances.

    This class is stateless; comparison methods take Snapshot objects or their state_dicts
    directly as arguments.
    """
    def __init__(self):
        """Initializes a Comparer object. It is stateless."""
        pass

    def L2_distance_state_dicts(self, state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Computes the L2 distance for each corresponding tensor in two state_dicts.

        If a parameter key is present in one state_dict but not the other, its L2 norm 
        in the present state_dict is reported as the distance (as if compared to zero).
        If parameter shapes or dtypes (after attempting cast) mismatch for the same key, 
        a distance of -1.0 is reported for that parameter, indicating an invalid comparison.

        Args:
            state_dict1 (Dict[str, torch.Tensor]): The first model state dictionary.
            state_dict2 (Dict[str, torch.Tensor]): The second model state dictionary.

        Returns:
            Dict[str, float]: A dictionary mapping each parameter name (key) to its 
                              L2 distance. A value of -1.0 indicates an issue with 
                              comparison for that specific parameter (e.g., shape mismatch).
        """
        distances = {}
        all_keys = set(state_dict1.keys()) | set(state_dict2.keys())

        for key in all_keys:
            if key not in state_dict1:
                distances[key] = torch.norm(state_dict2[key]).item()
                continue
            if key not in state_dict2:
                distances[key] = torch.norm(state_dict1[key]).item()
                continue

            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]

            if tensor1.shape != tensor2.shape:
                distances[key] = -1.0 # Indicator for shape mismatch
                continue
            
            if tensor1.dtype != tensor2.dtype:
                try:
                    tensor2 = tensor2.to(tensor1.dtype)
                except Exception:
                    distances[key] = -1.0 # Indicator for dtype mismatch/cast fail
                    continue

            distance = torch.norm(tensor1 - tensor2, p=2).item()
            distances[key] = distance
        return distances

    def total_L2_distance_state_dicts(self, state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> float:
        """
        Computes an aggregate L2 distance between all parameters in two state_dicts.

        This is calculated as the square root of the sum of squared L2 distances of 
        individual, validly comparable parameters. Parameters with comparison issues 
        (e.g., shape mismatches, indicated by -1.0 from `L2_distance_state_dicts`) are 
        excluded from this sum.

        Args:
            state_dict1 (Dict[str, torch.Tensor]): The first model state dictionary.
            state_dict2 (Dict[str, torch.Tensor]): The second model state dictionary.

        Returns:
            float: The aggregate L2 distance. Returns 0.0 if all validly compared 
                   parameters are identical. Returns -1.0 if no parameters could be 
                   validly compared (e.g., all parameters had mismatches).
        """
        param_distances = self.L2_distance_state_dicts(state_dict1, state_dict2)
        valid_distances_sq = [d**2 for d in param_distances.values() if d >= 0]
        
        if not param_distances: # Should not happen if state_dicts are not empty
            return 0.0 
        if not valid_distances_sq: 
            if all(d == 0 for d in param_distances.values() if d >=0 ): return 0.0
            if all(d < 0 for d in param_distances.values()): return -1.0
            return 0.0 # Or handle as an error if expected valid distances

        return sum(valid_distances_sq)**0.5

    def compare_snapshots(self, snapshot1: Snapshot, snapshot2: Snapshot, per_layer: bool = True) -> Dict[str, Any]:
        """
        Compares two Snapshot objects and returns a dictionary of comparison results.

        Args:
            snapshot1 (Snapshot): The first snapshot.
            snapshot2 (Snapshot): The second snapshot.
            per_layer (bool, optional): If True, computes and includes per-layer L2 distances 
                                      and an aggregate L2 distance derived from these. 
                                      If False, computes only a single total L2 distance. 
                                      Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing metadata from both snapshots, 
                            the comparison type (e.g., "L2_distance"), and the 
                            calculated distance(s). Structure depends on `per_layer`:
                            - If `per_layer` is True: includes "per_layer_L2_distance" (Dict) 
                              and "aggregate_L2_distance_from_layers" (float).
                            - If `per_layer` is False: includes "total_L2_distance" (float).
        """
        results = {
            "snapshot1_metadata": snapshot1.metadata,
            "snapshot2_metadata": snapshot2.metadata,
            "comparison_type": "L2_distance"
        }

        if per_layer:
            distances = self.L2_distance_state_dicts(snapshot1.model_state, snapshot2.model_state)
            results["per_layer_L2_distance"] = distances
            valid_distances_sq = [d**2 for d in distances.values() if d >= 0]
            
            current_aggregate_dist = -1.0 # Default if no valid distances
            if valid_distances_sq: # If there are any valid squared distances
                current_aggregate_dist = sum(valid_distances_sq)**0.5
            elif not distances: # No distances at all (empty state dicts?)
                 current_aggregate_dist = 0.0
            elif all(d == 0 for d in distances.values() if d >=0): # All valid distances are zero
                 current_aggregate_dist = 0.0
            # If distances exist but all are < 0 (invalid), current_aggregate_dist remains -1.0
            results["aggregate_L2_distance_from_layers"] = current_aggregate_dist

        else:
            total_distance = self.total_L2_distance_state_dicts(snapshot1.model_state, snapshot2.model_state)
            results["total_L2_distance"] = total_distance
            
        return results

    def compare_consecutive_snapshots(self, snapshots: List[Snapshot], per_layer: bool = True) -> List[Dict[str, Any]]:
        """
        Compares all consecutive snapshots in a provided list.

        Args:
            snapshots (List[Snapshot]): A list of `Snapshot` objects, ordered by time 
                                      or training progression.
            per_layer (bool, optional): Passed to `compare_snapshots` for each pair. 
                                      Defaults to True.

        Returns:
            List[Dict[str, Any]]: A list of comparison result dictionaries, one for each 
                                  consecutive pair of snapshots. Returns an empty list 
                                  if fewer than two snapshots are provided.
        """
        comparisons = []
        if len(snapshots) < 2:
            return comparisons
        
        for i in range(len(snapshots) - 1):
            snap1 = snapshots[i]
            snap2 = snapshots[i+1]
            comparison_result = self.compare_snapshots(snap1, snap2, per_layer=per_layer)
            comparisons.append(comparison_result)
        return comparisons

# Example Usage (for testing, primary example is run_analysis_example.py)
if __name__ == '__main__':
    import os
    import time
    try:
        from ..core.snapshot import Snapshot
    except ImportError:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from neuro_change_tracker.core.snapshot import Snapshot

    class SimpleMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 5)
            self.fc2 = torch.nn.Linear(5, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    print("[INFO] Initializing test: Creating dummy model and snapshots...")
    model1 = SimpleMLP()
    snap1_data = {k: v.clone().detach() for k,v in model1.state_dict().items()}
    meta1 = {"epoch": 0, "step": 0, "name": "initial"}
    snapshot1 = Snapshot(snap1_data, meta1)
    print(f"[INFO] Created Snapshot 0: {meta1}")

    time.sleep(0.01)
    model2 = SimpleMLP()
    with torch.no_grad():
        model2.fc1.weight.data += 0.1
        model2.fc2.bias.data -= 0.05
    snap2_data = {k: v.clone().detach() for k,v in model2.state_dict().items()}
    meta2 = {"epoch": 1, "step": 100, "name": "epoch_1"}
    snapshot2 = Snapshot(snap2_data, meta2)
    print(f"[INFO] Created Snapshot 1: {meta2}")

    time.sleep(0.01)
    model3 = SimpleMLP()
    with torch.no_grad():
        model3.fc1.weight.data += 0.2 
        model3.fc2.bias.data -= 0.1  
    snap3_data = {k: v.clone().detach() for k,v in model3.state_dict().items()}
    meta3 = {"epoch": 2, "step": 200, "name": "epoch_2"}
    snapshot3 = Snapshot(snap3_data, meta3)
    print(f"[INFO] Created Snapshot 2: {meta3}")

    snapshots_list = [snapshot1, snapshot2, snapshot3]

    # Initialize Comparer (now stateless)
    comparer = Comparer()
    print("[INFO] Comparer initialized.")

    print("\n[INFO] --- Comparing all consecutive snapshots (per_layer=True) ---")
    all_consecutive_comps = comparer.compare_consecutive_snapshots(snapshots_list, per_layer=True)
    
    if not all_consecutive_comps:
        print("[WARN] No consecutive comparisons generated.")
    
    for i, comp in enumerate(all_consecutive_comps):
        # Determine snap_idx based on available snapshots or comparison index
        snap_idx1 = i # Assuming comp[i] is snap[i] vs snap[i+1]
        snap_idx2 = i + 1
        print(f"\n[RESULT] Comparison {i}: Snapshot {snap_idx1} vs Snapshot {snap_idx2}")
        print(f"  Snapshot {snap_idx1} Metadata: {comp['snapshot1_metadata']}")
        print(f"  Snapshot {snap_idx2} Metadata: {comp['snapshot2_metadata']}")
        print(f"  Aggregate L2 Distance (from layers): {comp['aggregate_L2_distance_from_layers']:.4f}")
        print(f"  Per-layer L2 Distances:")
        if 'per_layer_L2_distance' in comp:
            for layer_name, dist in comp['per_layer_L2_distance'].items():
                print(f"    {layer_name}: {dist:.4f}")
        else:
            print("    Per-layer distances not available.")

    print("\n[INFO] --- Detailed comparison: Snapshot 0 and 2 (should show larger changes) ---")
    comp_0_2_per_layer = comparer.compare_snapshots(snapshots_list[0], snapshots_list[2], per_layer=True)
    print(f"[RESULT] Comparison: Snapshot 0 vs Snapshot 2")
    print(f"  Snapshot 0 Metadata: {comp_0_2_per_layer['snapshot1_metadata']}")
    print(f"  Snapshot 2 Metadata: {comp_0_2_per_layer['snapshot2_metadata']}")
    print(f"  Aggregate L2 Distance (from layers): {comp_0_2_per_layer['aggregate_L2_distance_from_layers']:.4f}")
    print(f"  Per-layer L2 Distances:")
    if 'per_layer_L2_distance' in comp_0_2_per_layer:
        for layer_name, dist in comp_0_2_per_layer['per_layer_L2_distance'].items():
            print(f"    {layer_name}: {dist:.4f}")

    print("\n[INFO] Test script finished.") 