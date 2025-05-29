import torch
import time
import os
from typing import Dict, Any

class Snapshot:
    def __init__(self, model_state: Dict[str, torch.Tensor], metadata: Dict[str, Any]):
        self.model_state = model_state
        self.metadata = metadata
        self.timestamp = time.time()

    def save(self, path: str, filename: str):
        """Saves the snapshot to a file using torch.save."""
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
        """Loads a snapshot from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")
            
        data = torch.load(filepath)
        # Reconstruct the Snapshot object
        snapshot = Snapshot(model_state=data['model_state_dict'], metadata=data['metadata'])
        snapshot.timestamp = data.get('timestamp', time.time()) # Handle older snapshots that might not have timestamp
        return snapshot 