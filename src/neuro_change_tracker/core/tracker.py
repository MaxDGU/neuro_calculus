import torch
import os
import datetime
import time # Added for the example usage, ensure it's here if example is kept or moved
from typing import Optional, Dict, Any, List

from .snapshot import Snapshot

class Tracker:
    """
    Manages the capturing and storage of model snapshots during a training process.

    The Tracker is initialized with a PyTorch model and a base directory for saving snapshots.
    Each run of the tracker (i.e., each Tracker instance) will create a new timestamped 
    subdirectory within the `save_dir` to store its snapshots, preventing overwrites from 
    previous runs.

    Attributes:
        model (torch.nn.Module): The PyTorch model to track.
        save_dir (str): The timestamped directory where snapshots for this tracker 
                        instance will be saved. This is a subdirectory of the `save_dir` 
                        passed to `__init__`.
        snapshots (List[Snapshot]): A list of `Snapshot` objects captured in memory by this tracker instance.
        snapshot_files (List[str]): A list of filepaths where the snapshots have been saved.
    """
    def __init__(self, model: torch.nn.Module, save_dir: str = "neuro_snapshots"):
        """
        Initializes a Tracker object.

        Args:
            model (torch.nn.Module): The PyTorch model whose weights will be tracked.
            save_dir (str, optional): The base directory where snapshot subdirectories 
                                      will be created. Defaults to "neuro_snapshots".
                                      A timestamped subdirectory will be created inside this 
                                      directory for each Tracker instance.
        """
        self.model = model
        # Create a unique subdirectory for this tracking session based on current time
        session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, session_timestamp)
        self.snapshots: List[Snapshot] = []
        self.snapshot_files: List[str] = []

        os.makedirs(self.save_dir, exist_ok=True)
        # print(f"Tracker initialized. Snapshots will be saved in: {self.save_dir}")

    def capture_snapshot(self, epoch: Optional[int] = None, step: Optional[int] = None, custom_metadata: Optional[Dict[str, Any]] = None):
        """
        Captures a snapshot of the model's current state_dict and saves it.

        The model's state_dict is deep-copied. Metadata including epoch, step,
        and any custom_metadata provided are stored with the snapshot.
        The snapshot is saved to a .pt file in the tracker's `save_dir`.

        Args:
            epoch (Optional[int], optional): Current epoch number. Defaults to None.
            step (Optional[int], optional): Current step number. Defaults to None.
            custom_metadata (Optional[Dict[str, Any]], optional): 
                Any additional custom metadata to store with the snapshot 
                (e.g., loss values, learning rate). Defaults to None.
        """
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
        snapshot_filename = f"snapshot_epoch_{epoch if epoch is not None else 'na'}_step_{step if step is not None else 'na'}_ts_{snapshot.timestamp:.0f}"
        
        snapshot.save(path=self.save_dir, filename=snapshot_filename)
        self.snapshot_files.append(os.path.join(self.save_dir, f"{snapshot_filename}.pt"))
        # print(f"Captured snapshot: {snapshot_filename}.pt")

    def get_last_snapshot(self) -> Optional[Snapshot]:
        """
        Returns the most recent snapshot captured by this tracker instance.

        Returns:
            Optional[Snapshot]: The last `Snapshot` object, or None if no snapshots 
                                have been captured.
        """
        return self.snapshots[-1] if self.snapshots else None

    def get_all_snapshots(self) -> List[Snapshot]:
        """
        Returns all snapshots captured by this tracker instance, in order of capture.

        Returns:
            List[Snapshot]: A list of all `Snapshot` objects.
        """
        return self.snapshots

    def get_snapshot_files(self) -> List[str]:
        """
        Returns the file paths of all snapshots saved by this tracker instance.

        Returns:
            List[str]: A list of full filepaths to the saved .pt snapshot files.
        """
        return self.snapshot_files

    # Example of how one might load snapshots if needed, though comparison might happen externally
    @classmethod
    def load_snapshots_from_dir(cls, directory: str) -> List[Snapshot]:
        """
        Loads all snapshots from a specified directory.

        This is a class method and can be used to load snapshots from any directory
        that contains .pt files saved by the Snapshot class, typically a `save_dir` 
        from a previous Tracker session.

        Args:
            directory (str): The directory path from which to load snapshots.

        Returns:
            List[Snapshot]: A list of loaded `Snapshot` objects, sorted by filename.
                          Returns an empty list if the directory is not found or 
                          contains no valid snapshot files.
        """
        loaded_snapshots = []
        if not os.path.isdir(directory):
            # print(f"Directory not found: {directory}")
            return loaded_snapshots
        
        # Sort files to maintain a somewhat predictable order, though timestamps in metadata are better for strict ordering
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

# Example usage (for testing purposes, primary example is run_analysis_example.py)
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
    # Test with a specific base directory for snapshots
    test_output_basedir = "test_tracker_output"
    tracker = Tracker(model, save_dir=test_output_basedir)
    print(f"Tracker initialized. Snapshots for this run will be in: {tracker.save_dir}")

    for epoch_num in range(3):
        with torch.no_grad():
            model.fc1.weight[0,0] = torch.tensor(float(epoch_num))
        
        tracker.capture_snapshot(epoch=epoch_num, step=(epoch_num+1)*100, custom_metadata={"loss": epoch_num * 0.1})
        time.sleep(0.01) 

    print(f"\nSnapshots captured by tracker instance: {len(tracker.get_all_snapshots())}")
    print(f"Snapshot files saved: {tracker.get_snapshot_files()}")

    if tracker.get_all_snapshots():
        last_snap = tracker.get_last_snapshot()
        if last_snap:
            print(f"\nLast snapshot metadata: {last_snap.metadata}")
            print(f"Last snapshot fc1.weight[0,0]: {last_snap.model_state['fc1.weight'][0,0]}")

    # Test loading from the specific directory this tracker instance used
    print(f"\nLoading snapshots directly from tracker's save directory: {tracker.save_dir}")
    loaded_directly = Tracker.load_snapshots_from_dir(tracker.save_dir)
    print(f"Total snapshots loaded from {tracker.save_dir}: {len(loaded_directly)}")
    if loaded_directly:
        print(f"First loaded snapshot (from dir) metadata: {loaded_directly[0].metadata}")
        print(f"  Weight fc1[0,0]: {loaded_directly[0].model_state['fc1.weight'][0,0]}")

    # Optional: cleanup the base test directory for multiple runs if desired
    # import shutil
    # if os.path.exists(test_output_basedir):
    #     shutil.rmtree(test_output_basedir)
    #     print(f"Cleaned up base test directory: {test_output_basedir}")

    # Cleanup test directory (optional)
    # import shutil
    # if os.path.exists("test_snapshots_output"): # Careful with shutil.rmtree
    #     shutil.rmtree("test_snapshots_output")
    #     print("Cleaned up test_snapshots_output directory.")
    # if os.path.exists(tracker.save_dir):
    #      shutil.rmtree(tracker.save_dir) # Be careful with this, especially if save_dir is not specific enough
    #      print(f"Cleaned up {tracker.save_dir} directory.") 