# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # import numpy as np
# # # from encoder import Encoder  # Import the Encoder class
# # # from decoder import Decoder
# # # from loss_camper import ChamferDistance
# # # from metric import l1_cd
# # # from loss import cd_loss_L1

# # # file_path = "/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/wireframes/1f343169e6948a2a5b7d8e48ecc58356.npy"
# # # point_cloud_data = np.load(file_path)  # Load the .npy file
# # # point_cloud_data=point_cloud_data.reshape(1,3,15000)
# # # input_tensor = torch.FloatTensor(point_cloud_data)  # Convert to float tensor
# # # # Initialize the model
# # # model = Encoder(num_points=15000)
# # # model.eval()
# # # with torch.no_grad():
# # #     features = model(input_tensor)
# # # decoder = Decoder(num_points=15000)
# # # final_pointcloud = decoder(features)

# # # # decoder.visualize_point_cloud(final_pointcloud)
# # # print("final point cloud,",final_pointcloud.shape)

# # # # Forward pass through the model

# # # # Print the output features
# # # print(features.shape)  # Should print the shape of the output features
# # # print(features)  # Optional: Print the feature values   
# # # file_path_sketch = "/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/ground_truth/1f343169e6948a2a5b7d8e48ecc58356.npy"
# # # point_cloud_data = np.load(file_path_sketch)  # Load the .npy file
# # # point_cloud_data1=point_cloud_data.reshape(1,3,15000)
# # # loss=cd_loss_L1(final_pointcloud,point_cloud_data1)
# # # print(loss)
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np
# # from encoder import Encoder  # Import the Encoder class
# # from decoder import Decoder
# # from loss_camper import ChamferDistance
# # from metric import l1_cd
# # from loss import cd_loss_L1 # Check if CUDA is available
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # file_path = "/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/wireframes/1f343169e6948a2a5b7d8e48ecc58356.npy"
# # point_cloud_data = np.load(file_path)  # Load the .npy file
# # point_cloud_data=point_cloud_data.reshape(1,3,15000)
# # input_tensor = torch.FloatTensor(point_cloud_data).to(device)  # Convert to float tensor and move to CUDA

# # # Initialize the model
# # model = Encoder(num_points=15000).to(device)  # Move the model to CUDA
# # model.eval()
# # with torch.no_grad():
# #     features = model(input_tensor)
# # decoder = Decoder(num_points=15000).to(device)  # Move the decoder to CUDA
# # final_pointcloud = decoder(features)

# # # decoder.visualize_point_cloud(final_pointcloud)
# # print("final point cloud,",final_pointcloud.shape)

# # # Forward pass through the model

# # # Print the output features
# # print(features.shape)  # Should print the shape of the output features
# # print(features)  # Optional: Print the feature values

# # file_path_sketch = "/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/ground_truth/1f343169e6948a2a5b7d8e48ecc58356.npy"
# # point_cloud_data1 = np.load(file_path_sketch)  # Load the .npy file
# # point_cloud_data1=point_cloud_data1.reshape(1,3,15000)
# # point_cloud_data1 = torch.FloatTensor(point_cloud_data1).to(device)  # Move the ground truth to CUDA
# # loss=cd_loss_L1(final_pointcloud,input_tensor)
# # print(loss)
# # import torch
# # import os
# # import numpy as np
# # from torch.utils.data import DataLoader
# # from torch.optim import Adam
# # from encoder import Encoder  # Import the Encoder class
# # from decoder import Decoder
# # from loss import cd_loss_L1  # Chamfer Distance Loss

# # # Custom Dataset Class for Loading Wireframe and Ground Truth Point Clouds
# # class PointCloudDataset(torch.utils.data.Dataset):
# #      def __init__(self, wireframe_dir, ground_truth_dir, transform=None):
# #          self.wireframe_dir = wireframe_dir
# #          self.ground_truth_dir = ground_truth_dir
# #          self.transform = transform

# # #         # List all files in the directories
# #          self.wireframe_files = sorted(os.listdir(wireframe_dir))
# #          self.ground_truth_files = sorted(os.listdir(ground_truth_dir))
        
# # #         # Ensure the number of wireframe and ground truth files match
# #          assert len(self.wireframe_files) == len(self.ground_truth_files), \
# #              "Mismatch between wireframe and ground truth data."

# #      def __len__(self):
# #          return len(self.wireframe_files)

# #      def __getitem__(self, idx):
# #          # Load wireframe and ground truth .npy files
# #          wireframe_path = os.path.join(self.wireframe_dir, self.wireframe_files[idx])
# #          ground_truth_path = os.path.join(self.ground_truth_dir, self.ground_truth_files[idx])
        
# #          wireframe = np.load(wireframe_path)
# #          ground_truth = np.load(ground_truth_path)

# #          # Apply transformations (e.g., data augmentation)
# #          if self.transform:
# #              wireframe, ground_truth = self.transform(wireframe, ground_truth)

# #          return torch.tensor(wireframe, dtype=torch.float32), torch.tensor(ground_truth, dtype=torch.float32)

# #  # Create DataLoader
# #      def create_dataloader(wireframe_dir, ground_truth_dir, batch_size=16, shuffle=True, num_workers=4, transform=None):
# #       dataset = PointCloudDataset(wireframe_dir, ground_truth_dir, transform=transform)
# #       dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
# #       return dataloader

# # # # Training function
# #      def train(model_encoder, model_decoder, dataloader, optimizer, device):
# #       model_encoder.train()
# #       model_decoder.train()
# #       total_loss = 0

# #       for batch_idx, (wireframes, ground_truths) in enumerate(dataloader):
# #          wireframes, ground_truths = wireframes.to(device), ground_truths.to(device)
# #          print(wireframes.shape)
# #          # Forward pass through Encoder and Decoder
# #          features = model_encoder(wireframes)
# #          final_pointcloud = model_decoder(features)

# # #         # Compute the loss
# #          loss = cd_loss_L1(final_pointcloud, ground_truths)

# # #         # Backpropagation
# #          optimizer.zero_grad()
# #          loss.backward()
# #          optimizer.step()

# #          total_loss += loss.item()

# #          avg_loss = total_loss / len(dataloader)
# #          return avg_loss

# #  # Validation or Testing function
# #      def evaluate(model_encoder, model_decoder, dataloader, device):
# #       model_encoder.eval()
# #       model_decoder.eval()
# #       total_loss = 0

# #       with torch.no_grad():
# #          for wireframes, ground_truths in dataloader:

# #              wireframes, ground_truths = wireframes.to(device), ground_truths.to(device)
# #              input_tensor = wireframes.transpose(1, 2)  # Transpose to shape [32, 3, 15000]

# # #             # Forward pass through Encoder and Decoder
# #              features = model_encoder(input_tensor)
# #              final_pointcloud = model_decoder(features)

# # #             # Compute the loss
# #              loss = cd_loss_L1(final_pointcloud, ground_truths)
# #              total_loss += loss.item()

# #       avg_loss = total_loss / len(dataloader)
# #       return avg_loss

# #      def main(num_epochs=10, batch_size=32, learning_rate=0.001):
# # #     # Directories containing train, validation, and test data
# #       train_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/train/wireframes'
# #       train_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/train/ground_truth'

# #       val_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/wireframes'
# #       val_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/ground_truth'

# #       test_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/test/wireframes'
# #       test_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/test/ground_truth'

# #      # Device configuration
# #       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # #     # Initialize Encoder and Decoder models
# #       model_encoder = Encoder(num_points=15000).to(device)
# #       model_decoder = Decoder(num_points=15000).to(device)

# # #     # Create data loaders
# #       train_loader = create_dataloader(train_wireframe_dir, train_ground_truth_dir, batch_size=batch_size, shuffle=True)
# #       val_loader = create_dataloader(val_wireframe_dir, val_ground_truth_dir, batch_size=batch_size, shuffle=False)
# #       test_loader = create_dataloader(test_wireframe_dir, test_ground_truth_dir, batch_size=batch_size, shuffle=False)

# # #     # Optimizer
# #      optimizer = Adam(list(model_encoder.parameters()) + list(model_decoder.parameters()), lr=learning_rate)

# # #     # Training loop
# #      for epoch in range(num_epochs):
# #          train_loss = train(model_encoder, model_decoder, train_loader, optimizer, device)
# #          val_loss = evaluate(model_encoder, model_decoder, val_loader, device)

# #          print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# # #     # Test after training
# #      test_loss = evaluate(model_encoder, model_decoder, test_loader, device)
# #      print(f"Test Loss: {test_loss:.4f}")

# # # if __name__ == '__main__':
# #      main(num_epochs=20)  # Example: training for 20 epochs
# import torch
# import time
# import open3d as o3d
# from torch.optim import Adam
# from encoder import Encoder  # Import the Encoder class
# from decoder import Decoder  # Import the Decoder class
# from loss import cd_loss_L1  # Import the Chamfer Distance Loss
# from dataset_maker import create_dataloader  # Import from datamaker
# def visualize_point_cloud(predicted_pointcloud, ground_truth_pointcloud):
#     """
#     Visualizes the predicted and ground truth point clouds using Open3D.
    
#     Args:
#         predicted_pointcloud (numpy array): The predicted point cloud (shape: [num_points, 3])
#         ground_truth_pointcloud (numpy array): The ground truth point cloud (shape: [num_points, 3])
#     """
#     # Convert predicted and ground truth point clouds to Open3D PointCloud format
#     pred_pc = o3d.geometry.PointCloud()
#     pred_pc.points = o3d.utility.Vector3dVector(predicted_pointcloud)

#     gt_pc = o3d.geometry.PointCloud()
#     gt_pc.points = o3d.utility.Vector3dVector(ground_truth_pointcloud)

#     # Color the predicted and ground truth point clouds differently
#     pred_pc.paint_uniform_color([1, 0, 0])  # Red for predicted point cloud
#     gt_pc.paint_uniform_color([0, 1, 0])    # Green for ground truth point cloud

#     # Visualize both point clouds
#     o3d.visualization.draw_geometries([pred_pc, gt_pc],
#                                       window_name="Predicted vs Ground Truth",
#                                       point_show_normal=False)
# # Training function
# def train(model_encoder, model_decoder, dataloader, optimizer, device):
#     model_encoder.train()
#     model_decoder.train()
#     total_loss = 0

#     for batch_idx, (wireframes, ground_truths) in enumerate(dataloader):
#         wireframes, ground_truths = wireframes.to(device), ground_truths.to(device)

#         # Transpose wireframes to shape [batch_size, 3, num_points] for conv1d
#         input_tensor = wireframes.transpose(1, 2)

#         # Forward pass through Encoder and Decoder
#         features = model_encoder(input_tensor)
#         final_pointcloud = model_decoder(features)

#         # Compute the loss
#         loss = cd_loss_L1(final_pointcloud, ground_truths)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# # Validation or Testing function
# def evaluate(model_encoder, model_decoder, dataloader, device,visualize=False):
#     model_encoder.eval()
#     model_decoder.eval()
#     total_loss = 0

#     with torch.no_grad():
#         for wireframes, ground_truths in dataloader:
#             wireframes, ground_truths = wireframes.to(device), ground_truths.to(device)

#             # Transpose wireframes to shape [batch_size, 3, num_points] for conv1d
#             input_tensor = wireframes.transpose(1, 2)
print
#             if visualize:
#                 # Visualize the first point cloud from the batch
#                 visualize_point_cloud(final_pointcloud[0].cpu().numpy(), ground_truths[0].cpu().numpy())

#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# def main(num_epochs=10, batch_size=32, learning_rate=0.001):
#     # Directories containing train, validation, and test data
#     train_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/train/wireframes'
#     train_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/train/ground_truth'

#     val_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/wireframes'
#     val_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/ground_truth'

#     test_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/test/wireframes'
#     test_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/test/ground_truth'

#     # Device configuration
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Initialize Encoder and Decoder models
#     model_encoder = Encoder(num_points=15000).to(device)
#     model_decoder = Decoder(num_points=15000).to(device)

#     # Create data loaders
#     train_loader = create_dataloader(train_wireframe_dir, train_ground_truth_dir, batch_size=batch_size, shuffle=True)
#     val_loader = create_dataloader(val_wireframe_dir, val_ground_truth_dir, batch_size=batch_size, shuffle=False)
#     test_loader = create_dataloader(test_wireframe_dir, test_ground_truth_dir, batch_size=batch_size, shuffle=False)

#     # Optimizer
#     optimizer = Adam(list(model_encoder.parameters()) + list(model_decoder.parameters()), lr=learning_rate)
#     total_start_time = time.time()
#     # Training loop
#     for epoch in range(num_epochs):
#         epoch_start_time = time.time()
#         train_loss = train(model_encoder, model_decoder, train_loader, optimizer, device)
#         val_loss = evaluate(model_encoder, model_decoder, val_loader, device)
#         epoch_end_time = time.time()
#         epoch_duration = epoch_end_time - epoch_start_time
        
#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#         print(f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds")
#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#     total_end_time = time.time()
#     total_training_time = total_end_time - total_start_time
#     print(f"Total training time for {num_epochs} epochs: {total_training_time:.2f} seconds")

#     # Test after training
#     test_loss = evaluate(model_encoder, model_decoder, test_loader, device,visualize=True)
#     print(f"Test Loss: {test_loss:.4f}")

# if __name__ == '__main__':
#     main(num_epochs=50)  # Example: training for 20 epochs
import time
import torch
import os
import open3d as o3d
from dataset_maker import PointCloudDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from encoder import Encoder  # Import the Encoder class
from decoder import Decoder
from loss import cd_loss_L1  # Chamfer Distance Loss
def visualize_point_cloud(predicted_pointcloud, ground_truth_pointcloud):
    """
    Visualizes the predicted and ground truth point clouds using Open3D.
    
    Args:
        predicted_pointcloud (numpy array): The predicted point cloud (shape: [num_points, 3])
        ground_truth_pointcloud (numpy array): The ground truth point cloud (shape: [num_points, 3])
    """
    # Convert predicted and ground truth point clouds to Open3D PointCloud format
    pred_pc = o3d.geometry.PointCloud()
    pred_pc.points = o3d.utility.Vector3dVector(predicted_pointcloud)

    gt_pc = o3d.geometry.PointCloud()
    gt_pc.points = o3d.utility.Vector3dVector(ground_truth_pointcloud)

    # Color the predicted and ground truth point clouds differently
    pred_pc.paint_uniform_color([1, 0, 0])  # Red for predicted point cloud
    gt_pc.paint_uniform_color([0, 1, 0])    # Green for ground truth point cloud

    # Visualize both point clouds
    o3d.visualization.draw_geometries([pred_pc, gt_pc],
                                      window_name="Predicted vs Ground Truth",
                                      point_show_normal=False)

def save_checkpoint(model_encoder, model_decoder, optimizer, epoch, loss, checkpoint_dir='checkpoints'):
    """Saves the model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_encoder_state_dict': model_encoder.state_dict(),
        'model_decoder_state_dict': model_decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

def create_dataloader(wireframe_dir, ground_truth_dir, batch_size=16, shuffle=True, num_workers=4, transform=None):
    dataset = PointCloudDataset(wireframe_dir, ground_truth_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def train(model_encoder, model_decoder, dataloader, optimizer, device):
    model_encoder.train()
    model_decoder.train()
    total_loss = 0

    for batch_idx, (wireframes, ground_truths) in enumerate(dataloader):
        wireframes, ground_truths = wireframes.to(device), ground_truths.to(device)
        input_tensor = wireframes.transpose(1, 2)

        # Forward pass through Encoder and Decoder
        features = model_encoder(input_tensor)
        final_pointcloud,densed_cloud = model_decoder(features)

        # Compute the loss
        loss = cd_loss_L1(densed_cloud, ground_truths)
         # Check loss tensor
        # print("Loss Tensor:", loss)
        # print("Loss grad_fn:", loss.grad_fn)  # Ensure it has a grad_fn

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model_encoder, model_decoder, dataloader, device, visualize=False):
    model_encoder.eval()
    model_decoder.eval()
    total_loss = 0

    with torch.no_grad():
        for wireframes, ground_truths in dataloader:
            wireframes, ground_truths = wireframes.to(device), ground_truths.to(device)
            input_tensor = wireframes.transpose(1, 2)  # Transpose to shape [batch_size, channels, length]

            # Forward pass through Encoder and Decoder
            features = model_encoder(input_tensor)
            final_pointcloud = model_decoder(features)
            final_pointcloud=final_pointcloud[0]
            # print(final_pointcloud.shape)
            # print(ground_truths.shape)
            if visualize:
                 # Visualize the first point cloud from the batch
                 visualize_point_cloud(final_pointcloud[0].cpu().numpy(), ground_truths[0].cpu().numpy())

            # Compute the loss
            loss = cd_loss_L1(final_pointcloud, ground_truths)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main(num_epochs=2000, batch_size=32, learning_rate=0.001):
    # Directories containing train, validation, and test data
    train_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/train/wireframes'
    train_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/train/ground_truth'

    val_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/wireframes'
    val_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/validate/ground_truth'

    test_wireframe_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/test/wireframes'
    test_ground_truth_dir = '/media/cvlab/EXT_HARD_DRIVE/lalit/project/main project/test/ground_truth'

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Encoder and Decoder models
    model_encoder = Encoder().to(device)
    model_decoder = Decoder().to(device)

    # Create data loaders
    train_loader = create_dataloader(train_wireframe_dir, train_ground_truth_dir, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_wireframe_dir, val_ground_truth_dir, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_wireframe_dir, test_ground_truth_dir, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = Adam(list(model_encoder.parameters()) + list(model_decoder.parameters()), lr=learning_rate)

    # Measure total training time
    total_start_time = time.time()

    # Training loop
    # for epoch in range(num_epochs):
    #     epoch_start_time = time.time()  # Measure time for each epoch
        
    #     train_loss = train(model_encoder, model_decoder, train_loader, optimizer, device)
    #     val_loss = evaluate(model_encoder, model_decoder, val_loader, device, visualize=False)
        
    #     epoch_end_time = time.time()
    #     epoch_duration = epoch_end_time - epoch_start_time
        
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    #     print(f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds")

    #     # Save checkpoint every 200 epochs
    #     if (epoch + 1) % 200 == 0:
    #         save_checkpoint(model_encoder, model_decoder, optimizer, epoch + 1, train_loss)

    # total_end_time = time.time()
    # total_training_time = total_end_time - total_start_time
    # print(f"Total training time for {num_epochs} epochs: {total_training_time:.2f} seconds")
    import wandb  # Import wandb at the beginning of your script

# Initialize WandB
    wandb.init(project="lalit123454")  # Specify your project name

# Start the training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Measure time for each epoch
    
    # Train and evaluate
        train_loss = train(model_encoder, model_decoder, train_loader, optimizer, device)
        val_loss = evaluate(model_encoder, model_decoder, val_loader, device, visualize=False)
    
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
    
    # Log the losses to WandB
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    # Print the training and validation loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds")

    # Save checkpoint every 200 epochs
        if (epoch + 1) % 200 == 0:
            save_checkpoint(model_encoder, model_decoder, optimizer, epoch + 1, train_loss)

# After training
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Total training time for {num_epochs} epochs: {total_training_time:.2f} seconds")

# Finish the WandB run
    wandb.finish()  # Make sure to call this at the end of your script
 
    # Test after training
    test_loss = evaluate(model_encoder, model_decoder, test_loader, device, visualize=True)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main(num_epochs=2000)  # Example: training for 1000 epochs
