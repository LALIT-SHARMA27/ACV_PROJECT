import torch
import os
import numpy as np
from torch.utils.data import DataLoader

# Custom Dataset Class for Loading Wireframe and Ground Truth Point Clouds
class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, wireframe_dir, ground_truth_dir, transform=None):
        self.wireframe_dir = wireframe_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform

        # List all files in the directories
        self.wireframe_files = sorted(os.listdir(wireframe_dir))
        self.ground_truth_files = sorted(os.listdir(ground_truth_dir))

        # Ensure the number of wireframe and ground truth files match
        assert len(self.wireframe_files) == len(self.ground_truth_files), \
            "Mismatch between wireframe and ground truth data."

    def __len__(self):
        return len(self.wireframe_files)

    def __getitem__(self, idx):
        # Load wireframe and ground truth .npy files
        wireframe_path = os.path.join(self.wireframe_dir, self.wireframe_files[idx])
        ground_truth_path = os.path.join(self.ground_truth_dir, self.ground_truth_files[idx])

        wireframe = np.load(wireframe_path)
        ground_truth = np.load(ground_truth_path)

        # Apply transformations (e.g., data augmentation)
        if self.transform:
            wireframe, ground_truth = self.transform(wireframe, ground_truth)

        return torch.tensor(wireframe, dtype=torch.float32), torch.tensor(ground_truth, dtype=torch.float32)

# Create DataLoader function
def create_dataloader(wireframe_dir, ground_truth_dir, batch_size=16, shuffle=True, num_workers=4, transform=None):
    dataset = PointCloudDataset(wireframe_dir, ground_truth_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
