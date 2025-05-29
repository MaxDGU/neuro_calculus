import torch
import os
import datetime
from typing import Optional, Dict, Any, List

from .snapshot import Snapshot # Assuming Snapshot is in the same directory or properly pathed

class Tracker:
    def __init__(self, model: torch.nn.Module, save_dir: str = "neuro_snapshots"):
        self.model = model
        self.save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.snapshots: List[Snapshot] = []
        self.snapshot_files: List[str] = []

        os.makedirs(self.save_dir, exist_ok=True)
        # print(f"Tracker initialized. Snapshots will be saved in: {self.save_dir}")

    def capture_snapshot(self, epoch: Optional[int] = None, step: Optional[int] = None, custom_metadata: Optional[Dict[str, Any]] = None):
        """Captures a snapshot of the model's current state."""
        model_state_dict = self.model.state_dict()
        # Create a deep copy of the state_dict to avoid issues if the model changes immediately after
        # This is important because state_dict() returns a reference in some PyTorch versions/scenarios.
        # For tensors, copy_() or clone().detach() is usually sufficient.
        copied_state_dict = {k: v.clone().detach() for k, v in model_state_dict.items()}

        metadata = {
            "epoch": epoch,
            "step": step,
        }
        if custom_metadata:
            metadata.update(custom_metadata)
        
        snapshot = Snapshot(model_state=copied_state_dict, metadata=metadata)
        self.snapshots.append(snapshot)
        
        # Define a filename strategy (e.g., based on epoch, step, or a counter)
        snapshot_filename = f"snapshot_epoch_{epoch or 'na'}_step_{step or 'na'}_ts_{snapshot.timestamp:.0f}"
        
        snapshot.save(path=self.save_dir, filename=snapshot_filename)
        self.snapshot_files.append(os.path.join(self.save_dir, f"{snapshot_filename}.pt"))
        # print(f"Captured snapshot: {snapshot_filename}.pt")

    def get_last_snapshot(self) -> Optional[Snapshot]:
        """Returns the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_all_snapshots(self) -> List[Snapshot]:
        """Returns all captured snapshots."""
        return self.snapshots

    def get_snapshot_files(self) -> List[str]:
        """Returns the file paths of all saved snapshots."""
        return self.snapshot_files

    # Example of how one might load snapshots if needed, though comparison might happen externally
    @classmethod
    def load_snapshots_from_dir(cls, directory: str) -> List[Snapshot]:
        """Loads all snapshots from a specified directory."""
        loaded_snapshots = []
        if not os.path.isdir(directory):
            # print(f"Directory not found: {directory}")
            return loaded_snapshots
        
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pt"):
                filepath = os.path.join(directory, filename)
                try:
                    snapshot = Snapshot.load(filepath)
                    loaded_snapshots.append(snapshot)
                    # print(f"Loaded snapshot from {filepath}")
                except Exception as e:
                    print(f"Error loading snapshot from {filepath}: {e}") # Or use a proper logger
        return loaded_snapshots

# Example usage (for testing purposes, would be in a separate script/notebook)
if __name__ == '__main__':
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

    model = SimpleMLP()
    tracker = Tracker(model, save_dir="test_snapshots_output")

    # Simulate a training loop
    for epoch_num in range(3):
        # Simulate some training that changes model weights
        # For this test, we'll just manually change a weight to see distinct snapshots
        with torch.no_grad():
            model.fc1.weight[0,0] = torch.tensor(float(epoch_num))
        
        tracker.capture_snapshot(epoch=epoch_num, step=(epoch_num+1)*100)
        time.sleep(0.1) # Ensure timestamps are different if saving rapidly

    print("Snapshots captured.")
    all_snaps = tracker.get_all_snapshots()
    print(f"Total snapshots: {len(all_snaps)}")
    for i, snap_file in enumerate(tracker.get_snapshot_files()):
        print(f"Snapshot file {i}: {snap_file}")
        loaded_snap = Snapshot.load(snap_file)
        print(f"  Epoch: {loaded_snap.metadata.get('epoch')}, Step: {loaded_snap.metadata.get('step')}")
        print(f"  Weight fc1[0,0]: {loaded_snap.model_state['fc1.weight'][0,0]}")

    # Test loading from directory
    print("\nTesting loading snapshots from directory...")
    # Assuming the tracker saved to a subdirectory named by timestamp, we need to find it or pass it directly
    # For this example, let's use the tracker's save_dir property
    loaded_from_dir = Tracker.load_snapshots_from_dir(tracker.save_dir)
    print(f"Total snapshots loaded from dir: {len(loaded_from_dir)}")
    if loaded_from_dir:
        print(f"First loaded snapshot metadata: {loaded_from_dir[0].metadata}")
        print(f"  Weight fc1[0,0]: {loaded_from_dir[0].model_state['fc1.weight'][0,0]}")

    # Cleanup test directory (optional)
    # import shutil
    # if os.path.exists("test_snapshots_output"): # Careful with shutil.rmtree
    #     shutil.rmtree("test_snapshots_output")
    #     print("Cleaned up test_snapshots_output directory.")
    # if os.path.exists(tracker.save_dir):
    #      shutil.rmtree(tracker.save_dir) # Be careful with this, especially if save_dir is not specific enough
    #      print(f"Cleaned up {tracker.save_dir} directory.") 