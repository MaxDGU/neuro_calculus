import torch
import time
import os
from typing import Dict, Any

class Snapshot:
    """
    Represents a single snapshot of a model's state and associated metadata.

    Attributes:
        model_state (Dict[str, torch.Tensor]): The state dictionary of the model.
        metadata (Dict[str, Any]): Additional metadata associated with the snapshot 
                                   (e.g., epoch, step, loss).
        timestamp (float): The time when the snapshot was created (Unix timestamp).
    """
    def __init__(self, model_state: Dict[str, torch.Tensor], metadata: Dict[str, Any]):
        """
        Initializes a Snapshot object.

        Args:
            model_state (Dict[str, torch.Tensor]): The state dictionary of the model.
            metadata (Dict[str, Any]): Additional metadata for the snapshot.
        """
        self.model_state = model_state
        self.metadata = metadata
        self.timestamp = time.time()

    def save(self, path: str, filename: str):
        """
        Saves the snapshot to a file using torch.save.

        The snapshot is saved as a dictionary containing the model_state_dict,
        metadata, and timestamp.

        Args:
            path (str): The directory path where the snapshot file will be saved.
            filename (str): The name of the file (without extension, '.pt' will be added).
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{filename}.pt")
        
        data_to_save = {
            'model_state_dict': self.model_state,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
        torch.save(data_to_save, filepath)
        # print(f"Snapshot saved to {filepath}") # Logging can be handled by the Tracker or higher up

    @staticmethod
    def load(filepath: str) -> 'Snapshot':
        """
        Loads a snapshot from a file.

        Args:
            filepath (str): The full path to the snapshot file (.pt).

        Returns:
            Snapshot: The loaded Snapshot object.

        Raises:
            FileNotFoundError: If the snapshot file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")
            
        data = torch.load(filepath)
        # Reconstruct the Snapshot object
        snapshot = Snapshot(model_state=data['model_state_dict'], metadata=data['metadata'])
        snapshot.timestamp = data.get('timestamp', time.time()) # Handle older snapshots that might not have timestamp
        return snapshot 