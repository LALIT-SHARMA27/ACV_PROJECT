import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# T-net (Spatial Transformer Network)
class Tnet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension '''
    def __init__(self, dim, num_points=15000):
        super(Tnet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim 

        self.conv1 = nn.Conv1d(dim, 128, kernel_size=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)
        

    def forward(self, x):
        bs = x.shape[0]

        # pass through shared MLP layers (conv1d)
        x = self.bn1(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.bn2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.bn3(F.relu(self.conv3(x)))
        # print(x.shape)
        # max pool over num points
        x = self.max_pool(x).view(bs, -1)
        # print(x.shape)
        # pass through MLP
        x = self.bn4(F.relu(self.linear1(x)))
        # print(x.shape)
        x = self.bn5(F.relu(self.linear2(x)))
        # print(x.shape)
        x = self.linear3(x)
        # print(x.shape)
        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden
        # print(x.shape)
        return x


# Point Net Backbone (main Architecture)
class Encoder(nn.Module):
    def __init__(self, num_points=15000, num_global_feats=1024, local_feat=True):
        ''' Initializers:
                num_points - number of points in point cloud
                num_global_feats - number of Global Features for the main 
                                   Max Pooling layer
                local_feat - if True, forward() returns the concatenation 
                             of the local and global features
            '''
        super(Encoder, self).__init__()

        # if true concat local and global features
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        # print(self.tnet1)
        self.tnet2 = Tnet(dim=64, num_points=num_points)
        # print(self.tnet2)
        # shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)
        
        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    
    def forward(self, x):

        # get batch size
        bs = x.shape[0]
        
        # pass through first Tnet to get transform matrix
        A_input = self.tnet1(x)
        # print("transform matrix",A_input)
        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        # print(x.shape)
        # pass through first shared MLP
        x = self.bn1(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.bn2(F.relu(self.conv2(x)))
        # print(x.shape)
        # get feature transform
        A_feat = self.tnet2(x)
        # print("feature transform",A_feat)
        # perform second transformation across each (64 dim) feature in the batch
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        # print(x.shape)
        # store local point features for segmentation head
        local_features = x.clone()
        print("local_features",local_features.shape)
        # print("local dd",local_features.shape)
        # pass through second MLP
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        # print(x.shape)
        # get global feature vector and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        # print("global featuer",global_features.shape)
        critical_indexes = critical_indexes.view(bs, -1)
        features = torch.cat((local_features, 
                                   global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), 
                                    dim=1)
        print("features",features.shape)
        # print("global_features",global_features)
        return features

# Assuming the model is defined above as PointNetBackbone

# Example usage

# Forward pass through the encoder and decoder
# with torch.no_grad():
#     input_tensor = torch.FloatTensor(padded_data.reshape(1, 3, 15000))  # Your input data
#     completed_point_cloud = model(input_tensor)

# Visualize the completed point cloud

# Load the point cloud from a .npy file
