import torch
from typing import Dict, List, Any, Optional
from ..core.snapshot import Snapshot # Use relative import

class Comparer:
    def __init__(self, snapshots: Optional[List[Snapshot]] = None):
        self.snapshots = snapshots if snapshots else []

    def add_snapshot(self, snapshot: Snapshot):
        self.snapshots.append(snapshot)
        # Optionally sort by a metadata field like timestamp or step if order is critical
        # self.snapshots.sort(key=lambda s: s.timestamp)

    def set_snapshots(self, snapshots: List[Snapshot]):
        self.snapshots = snapshots
        # self.snapshots.sort(key=lambda s: s.timestamp)


    def L2_distance_state_dicts(self, state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Computes the L2 distance for each corresponding tensor in two state_dicts.
        Returns a dictionary энергия_parameter_name: L2_distance_value.
        """
        distances = {}
        all_keys = set(state_dict1.keys()) | set(state_dict2.keys()) # Union of all keys

        for key in all_keys:
            if key not in state_dict1:
                # print(f"Warning: Key {key} not found in first state_dict. Treating as all zeros for comparison.")
                # Or handle as an error, or use norm of tensor in state_dict2
                distances[key] = torch.norm(state_dict2[key]).item()
                continue
            if key not in state_dict2:
                # print(f"Warning: Key {key} not found in second state_dict. Treating as all zeros for comparison.")
                distances[key] = torch.norm(state_dict1[key]).item()
                continue

            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]

            if tensor1.shape != tensor2.shape:
                # print(f"Warning: Shapes for {key} differ: {tensor1.shape} vs {tensor2.shape}. Skipping.")
                distances[key] = -1.0 # Or raise an error, or some other indicator
                continue
            
            if tensor1.dtype != tensor2.dtype:
                # Attempt to cast or warn. For L2, dtypes should ideally match or be compatible numerics.
                # print(f"Warning: Dtypes for {key} differ: {tensor1.dtype} vs {tensor2.dtype}. Attempting comparison.")
                try:
                    tensor2 = tensor2.to(tensor1.dtype)
                except Exception as e:
                    # print(f"Could not cast dtype for {key}: {e}. Skipping.")
                    distances[key] = -1.0
                    continue

            distance = torch.norm(tensor1 - tensor2, p=2).item()
            distances[key] = distance
        return distances

    def total_L2_distance_state_dicts(self, state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> float:
        """
        Computes the total L2 distance between all parameters in two state_dicts.
        This is done by flattening all parameters into a single vector for each state_dict and then computing L2.
        Assumes all keys present in state_dict1 are also in state_dict2 and vice-versa, and have same shapes.
        For a more robust version, use L2_distance_state_dicts and sum the squares, then sqrt.
        """
        # A simple approach: concatenate all parameters into a single vector
        # This might be memory intensive for large models if not careful.
        vec1 = torch.cat([p.flatten() for p in state_dict1.values()])
        vec2 = torch.cat([p.flatten() for p in state_dict2.values()])
        
        if vec1.shape != vec2.shape:
            # This case should be handled by checking keys and shapes individually
            # print("Warning: Overall parameter vector shapes differ. This indicates missing/extra parameters or shape mismatches.")
            # Fallback to summing individual distances if shapes mismatch this way (which implies structural diff)
            param_distances = self.L2_distance_state_dicts(state_dict1, state_dict2)
            return sum(d**2 for d in param_distances.values() if d >= 0)**0.5


        return torch.norm(vec1 - vec2, p=2).item()


    def compare_snapshots(self, snapshot1_idx: int, snapshot2_idx: int, per_layer: bool = True) -> Dict[str, Any]:
        """
        Compares two snapshots by their indices in the internal list.
        Returns a dictionary with comparison results.
        """
        if not (0 <= snapshot1_idx < len(self.snapshots) and 0 <= snapshot2_idx < len(self.snapshots)):
            raise IndexError("Snapshot index out of bounds.")

        snap1 = self.snapshots[snapshot1_idx]
        snap2 = self.snapshots[snapshot2_idx]

        results = {
            "snapshot1_metadata": snap1.metadata,
            "snapshot2_metadata": snap2.metadata,
            "comparison_type": "L2_distance"
        }

        if per_layer:
            distances = self.L2_distance_state_dicts(snap1.model_state, snap2.model_state)
            results["per_layer_L2_distance"] = distances
            # Calculate an aggregate L2 norm from per-layer distances (sqrt of sum of squares)
            # This is more robust if layers can be added/removed or shapes change,
            # as total_L2_distance_state_dicts might fail or give misleading results.
            valid_distances_sq = [d**2 for d in distances.values() if d >= 0] # d >= 0 to filter out -1.0 for mismatches
            results["aggregate_L2_distance_from_layers"] = sum(valid_distances_sq)**0.5
        else:
            # This simpler total L2 assumes model structure is identical.
            total_distance = self.total_L2_distance_state_dicts(snap1.model_state, snap2.model_state)
            results["total_L2_distance"] = total_distance
            
        return results

    def compare_consecutive_snapshots(self, per_layer: bool = True) -> List[Dict[str, Any]]:
        """
        Compares all consecutive snapshots in the list.
        """
        comparisons = []
        if len(self.snapshots) < 2:
            # print("Not enough snapshots to compare consecutively.")
            return comparisons
        
        for i in range(len(self.snapshots) - 1):
            try:
                comparison_result = self.compare_snapshots(i, i + 1, per_layer=per_layer)
                comparisons.append(comparison_result)
            except IndexError as e: # Should not happen with the loop range
                # print(f"Error comparing snapshot {i} and {i+1}: {e}")
                pass 
        return comparisons

# Example Usage (for testing, to be moved to a test script or notebook)
if __name__ == '__main__':
    import os
    import time
    # Dummy model for testing
    class SimpleMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 5)
            self.fc2 = torch.nn.Linear(5, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create some dummy snapshots
    model1 = SimpleMLP()
    snap1_data = {k: v.clone().detach() for k,v in model1.state_dict().items()}
    meta1 = {"epoch": 0, "step": 0, "name": "initial"}
    snapshot1 = Snapshot(snap1_data, meta1)

    time.sleep(0.1) # ensure different timestamps

    model2 = SimpleMLP() # Fresh model, same architecture
    # Change weights for model2 to simulate training
    with torch.no_grad():
        model2.fc1.weight.data += 0.1
        model2.fc2.bias.data -= 0.05
    
    snap2_data = {k: v.clone().detach() for k,v in model2.state_dict().items()}
    meta2 = {"epoch": 1, "step": 100, "name": "epoch_1"}
    snapshot2 = Snapshot(snap2_data, meta2)

    time.sleep(0.1)

    model3 = SimpleMLP()
    with torch.no_grad():
        model3.fc1.weight.data += 0.2
        model3.fc2.bias.data -= 0.1
    snap3_data = {k: v.clone().detach() for k,v in model3.state_dict().items()}
    meta3 = {"epoch": 2, "step": 200, "name": "epoch_2"}
    snapshot3 = Snapshot(snap3_data, meta3)

    # Initialize Comparer
    comparer = Comparer()
    comparer.add_snapshot(snapshot1)
    comparer.add_snapshot(snapshot2)
    comparer.add_snapshot(snapshot3)

    print("--- Comparing snapshot 0 and 1 (per_layer=True) ---")
    comp_0_1_per_layer = comparer.compare_snapshots(0, 1, per_layer=True)
    print(f"Snapshot 1 Meta: {comp_0_1_per_layer['snapshot1_metadata']}")
    print(f"Snapshot 2 Meta: {comp_0_1_per_layer['snapshot2_metadata']}")
    print(f"Per-layer L2 distances: {comp_0_1_per_layer['per_layer_L2_distance']}")
    print(f"Aggregate L2 from layers: {comp_0_1_per_layer['aggregate_L2_distance_from_layers']}")

    print("\n--- Comparing snapshot 0 and 1 (per_layer=False) ---")
    comp_0_1_total = comparer.compare_snapshots(0, 1, per_layer=False)
    print(f"Total L2 distance: {comp_0_1_total['total_L2_distance']}")

    print("\n--- Comparing all consecutive snapshots (per_layer=True) ---")
    all_consecutive_comps = comparer.compare_consecutive_snapshots(per_layer=True)
    for i, comp in enumerate(all_consecutive_comps):
        print(f"Comparison {i} (Snap {i} vs Snap {i+1}):")
        print(f"  Aggregate L2: {comp['aggregate_L2_distance_from_layers']}")
        # print(f"  Layer diffs: {comp['per_layer_L2_distance']}")


    # Test with state dicts that have different keys (e.g. a layer was added/removed)
    # This is a more complex scenario that the current L2_distance_state_dicts handles by warning
    # and calculating norm for non-overlapping keys.
    print("\n--- Testing comparison with differing state dict keys ---")
    state_dict_a = {'layer1.weight': torch.randn(5,5), 'layer2.weight': torch.randn(3,3)}
    state_dict_b = {'layer1.weight': torch.randn(5,5) + 0.5, 'layer3.weight': torch.randn(2,2)}
    
    layer_diffs_ab = comparer.L2_distance_state_dicts(state_dict_a, state_dict_b)
    print(f"Layer diffs for A vs B (structural change): {layer_diffs_ab}")
    # The total_L2_distance_state_dicts would fail here if not careful because torch.cat requires same num elements
    # My implementation falls back to summing squares from L2_distance_state_dicts
    total_diff_ab = comparer.total_L2_distance_state_dicts(state_dict_a, state_dict_b)
    print(f"Total L2 for A vs B (structural change): {total_diff_ab}")
    
    # Test what happens if a layer's shape changes (e.g. fc layer units change)
    print("\n--- Testing comparison with differing tensor shapes for the same key ---")
    state_dict_c = {'fc.weight': torch.randn(10,5)}
    state_dict_d = {'fc.weight': torch.randn(10,8)} # Different shape
    layer_diffs_cd = comparer.L2_distance_state_dicts(state_dict_c, state_dict_d)
    print(f"Layer diffs for C vs D (shape change): {layer_diffs_cd}") # Will show -1.0 for fc.weight
    total_diff_cd = comparer.total_L2_distance_state_dicts(state_dict_c, state_dict_d)
    print(f"Total L2 for C vs D (shape change): {total_diff_cd}") 