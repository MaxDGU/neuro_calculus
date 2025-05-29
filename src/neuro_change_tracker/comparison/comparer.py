import torch
from typing import Dict, List, Any, Tuple
from ..core.snapshot import Snapshot # Use relative import

class Comparer:
    def __init__(self):
        """Comparer is now stateless regarding a list of snapshots."""
        pass

    def L2_distance_state_dicts(self, state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Computes the L2 distance for each corresponding tensor in two state_dicts.
        Returns a dictionary parameter_name: L2_distance_value.
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
        Computes the total L2 distance between all parameters in two state_dicts.
        This is done by flattening all parameters into a single vector for each state_dict and then computing L2.
        Handles potential mismatches in keys or shapes by summing squared valid per-layer distances.
        """
        param_distances = self.L2_distance_state_dicts(state_dict1, state_dict2)
        valid_distances_sq = [d**2 for d in param_distances.values() if d >= 0]
        
        if not valid_distances_sq: # Handles cases where all comparisons failed (e.g. all shapes mismatch)
             # Or if all params are identical across completely different architectures (unlikely but mathematically possible)
            if all(d == 0 for d in param_distances.values() if d >=0 ): return 0.0
            # If all are -1, it means no valid comparison could be made.
            if all(d < 0 for d in param_distances.values()): return -1.0 # Or raise error

        return sum(valid_distances_sq)**0.5

    def compare_snapshots(self, snapshot1: Snapshot, snapshot2: Snapshot, per_layer: bool = True) -> Dict[str, Any]:
        """
        Compares two Snapshot objects.
        Returns a dictionary with comparison results.
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
            results["aggregate_L2_distance_from_layers"] = sum(valid_distances_sq)**0.5 if valid_distances_sq else 0.0
            if all(d < 0 for d in distances.values()): # If all were invalid
                 results["aggregate_L2_distance_from_layers"] = -1.0

        else:
            # This simpler total L2 assumes model structure is identical for concatenation approach.
            # The refactored total_L2_distance_state_dicts is more robust now.
            total_distance = self.total_L2_distance_state_dicts(snapshot1.model_state, snapshot2.model_state)
            results["total_L2_distance"] = total_distance
            
        return results

    def compare_consecutive_snapshots(self, snapshots: List[Snapshot], per_layer: bool = True) -> List[Dict[str, Any]]:
        """
        Compares all consecutive snapshots in the provided list.
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

# Example Usage (for testing, to be moved to a test script or notebook)
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