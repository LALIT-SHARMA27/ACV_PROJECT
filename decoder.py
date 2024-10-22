import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
# class Decoder(nn.Module):
#     def __init__(self, features, num_points=15000):
#         super(Decoder, self).__init__()
        
#         self.num_points = num_points
#         self.num_global_feats = features
        
#         # Define the decoder network
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 1024)
#         self.fc3 = nn.Linear(1024, 2048)
#         self.fc4 = nn.Linear(2048, num_points * 3)  # Output will be 3 coordinates per point

#     def forward(self, features):
#         # Forward pass through the decoder
#         x = F.relu(self.fc1(features))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)

#         # Reshape to (batch_size, num_points, 3)
#         point_cloud = x.view(-1, self.num_points, 3)
#         return point_cloud

#     def visualize_point_cloud(self, point_cloud):
#         """Visualize the point cloud using Open3D."""
#         # Convert the point cloud to numpy array
#         point_cloud_np = point_cloud.detach().cpu().numpy().reshape(-1, 3)

#         # Create an Open3D point cloud object
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

#         # Visualize the point cloud
#         o3d.visualization.draw_geometries([pcd], window_name="Completed Point Cloud")
# class Decoder(nn.Module):
#     def __init__(self, num_points=15000):
#         super(Decoder, self).__init__()
        
#         self.num_points = num_points
        
#         # Define the decoder network
#         self.fc1 = nn.Linear(1024, 512)  # Assuming global feature has 1024 dimensions
#         self.fc2 = nn.Linear(512, 1024)
#         self.fc3 = nn.Linear(1024, 2048)
#         self.fc4 = nn.Linear(2048, num_points * 3)  # Output will be 3 coordinates per point

#     def forward(self, global_feature):
#         # Forward pass through the decoder
#         x = F.relu(self.fc1(global_feature))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)

#         # Reshape to (batch_size, num_points, 3)
#         point_cloud = x.view(-1, self.num_points, 3)
#         return point_cloud

#     def visualize_point_cloud(self, point_cloud):
#         """Visualize the point cloud using Open3D."""
#         # Convert the point cloud to numpy array
#         point_cloud_np = point_cloud.detach().cpu().numpy().reshape(-1, 3)

#         # Create an Open3D point cloud object
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

#         # Visualize the point cloud
#         o3d.visualization.draw_geometries([pcd], window_name="Completed Point Cloud")
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_coarse=1, grid_size=4):
        super(Decoder, self).__init__()
        
        self.num_coarse = num_coarse  # Number of coarse points
        self.grid_size = grid_size    # Size of the grid for fine point generation
        self.num_fine = num_coarse * grid_size * grid_size  # Number of fine points
        
        # Decoder MLP to produce the coarse point cloud
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)  # Generate num_coarse * 3 coordinates
        
        # Folding seed grid (grid_size x grid_size)
        self.folding_seed = self.create_folding_seed(grid_size).cuda()  # Move seed to GPU

        # Final convolution layers for fine point cloud generation
        self.final_conv = nn.Sequential(
            nn.Conv1d(1029, 512, 1),  # 1024 (global) + 3 (coarse points) + 2 (seed)
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 3, 1)  # Output 3 coordinates per point
        )

    def create_folding_seed(self, grid_size):
        """Creates the 2D grid seed used for folding."""
        # Create a uniform grid [-0.5, 0.5] x [-0.5, 0.5] with size grid_size
        grid = torch.meshgrid(
            torch.linspace(-0.5, 0.5, grid_size),
            torch.linspace(-0.5, 0.5, grid_size)
        )
        grid = torch.stack(grid, dim=-1).reshape(-1, 2)  # Shape (S, 2)
        return grid

    def forward(self, global_feature):
        """
        Input: global_feature of shape (B, 1024)
        Output: coarse point cloud of shape (B, num_coarse, 3)
                fine point cloud of shape (B, num_fine, 3)
        """
        B = global_feature.size(0)
        # print("B: ", B)
        # Ensure global_feature is on the GPU
        global_feature = global_feature.cuda()
        # print("global_feature: ", global_feature)
        # Step 1: Generate coarse point cloud
        x = F.relu(self.fc1(global_feature))
        # print("x: ", x)
        x = F.relu(self.fc2(x))
        # print("x2: ", x)
        coarse = self.fc3(x).view(B, self.num_coarse, 3)  # (B, num_coarse, 3)
        # print("coarse: ", coarse)
        # Step 2: Expand coarse points to prepare for fine generation
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)
        # print("point_feat: ", point_feat)
        point_feat = point_feat.reshape(B, self.num_fine, 3).transpose(2, 1)  # (B, 3, num_fine)
        # print("point_feat2: ", point_feat.shape)
        seed = self.folding_seed.unsqueeze(0).expand(B, -1, -1).cuda()  # (B, 16, 2)
    
        # Step 3: Expand the seed grid (ensure seed is on GPU)
        seed = seed.unsqueeze(2).repeat(1, 1, self.num_fine // (self.grid_size ** 2), 1)  # (B, 16, 1024, 2)
        seed = seed.view(B, 2, self.num_fine)
        # seed = self.folding_seed.unsqueeze(0).expand(B, -1, -1).cuda()  # (B, num_fine, 2)
        # print("seed: ", seed)
        # seed = seed.transpose(2, 1)  # (B, 2, num_fine)
        # print("seed2: ", seed.shape)
        # Step 4: Expand global feature to match fine points
        global_expanded = global_feature.unsqueeze(2).expand(-1, -1, self.num_fine)  # (B, 1024, num_fine)
        # print("global_expanded: ")
        # Step 5: Concatenate global feature, seed, and coarse point features
        feat = torch.cat([global_expanded, seed, point_feat], dim=1)  # (B, 1029, num_fine)
        # print("feat: ", feat)
        # Step 6: Apply final convolution to generate fine point cloud
        fine = self.final_conv(feat) + point_feat  # Add point_feat for refinement (B, 3, num_fine)
        # print("fine: ", fine)
        # print("coarse.cont: ", coarse.contiguous(), "fine.transpose: ", fine.transpose(1, 2).contiguous())
        # Return the coarse and fine point clouds (ensure they are contiguous for CUDA efficiency)
        return coarse.contiguous(), fine.transpose(1, 2).contiguous()

    def visualize_point_cloud(self, point_cloud):
        """Visualize the point cloud using Open3D."""
        # Convert the point cloud to numpy array
        point_cloud_np = point_cloud.detach().cpu().numpy().reshape(-1, 3)

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd], window_name="Completed Point Cloud")
