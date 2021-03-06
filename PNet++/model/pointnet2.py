import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model.pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from torch.nn import ConvTranspose2d, BatchNorm2d, ReLU
from model.convlstm import ConvLSTM2D

class PointNet2ClsMsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsMsg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x,l3_points


class PointNet2ClsSsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsSsg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x


class PointNet2PartSeg(nn.Module): #TODO part segmentation tasks
    def __init__(self, num_classes):
        super(PointNet2PartSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, feat

class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 0+3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)
        self.lstm = nn.LSTM(32,256)
        
        self.ct1 = ConvTranspose2d(1,8, (3,3))
        self.b1 = BatchNorm2d(8)

        self.ct2 = ConvTranspose2d(8,16, (3,3))
        self.b2 = BatchNorm2d(16)

        self.ct3 = ConvTranspose2d(16,32, (3,3))
        self.b3 = BatchNorm2d(32)
        self.ct4 = ConvTranspose2d(32,32, (3,3))
        self.b4 = BatchNorm2d(32)
        self.ct5 = ConvTranspose2d(32,32, (3,3))
        self.b5 = BatchNorm2d(32)
        self.ct6 = ConvTranspose2d(32,64, (3,3))
        self.b6 = BatchNorm2d(64)
        self.ct7 = ConvTranspose2d(64,64, (3,3))
        self.b7 = BatchNorm2d(64)
        self.ct8 = ConvTranspose2d(64,64, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.b8 = BatchNorm2d(64)
        self.ct9 = ConvTranspose2d(64,64, (3,3))
        self.b9 = BatchNorm2d(64)
        self.ct10 = ConvTranspose2d(64,128, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.b10 = BatchNorm2d(128)
        self.ct11 = ConvTranspose2d(128,128, (3,3))
        self.b11 = BatchNorm2d(128)
        self.ct12 = ConvTranspose2d(128,64, (3,3),stride=(4,2),padding=(0,1),output_padding=(1,1))
        self.b12 = BatchNorm2d(64)
        self.ct13 = ConvTranspose2d(64,32, (3,3))
        self.b13 = BatchNorm2d(32)
        self.ct14 = ConvTranspose2d(32,32, (3,3))
        self.b14 = BatchNorm2d(32)
        self.ct15 = ConvTranspose2d(32,32, (3,1))
        self.b15 = BatchNorm2d(32)
        self.ct16 = ConvTranspose2d(32,18, (3,1))
        self.b16 = BatchNorm2d(18)
        self.ct17 = ConvTranspose2d(18,18, (1,1))
        self.b17 = BatchNorm2d(18)
        self.ct18 = ConvTranspose2d(18,16, (1,1))
        self.b18= BatchNorm2d(16)
        self.ct19 = ConvTranspose2d(16,8, (1,1))
        self.b19 = BatchNorm2d(8)
        self.ct20 = ConvTranspose2d(8,1, (1,1))

    def forward(self, xyz, norm_plt):
        # Set Abstraction layers
       
        l1_xyz, l1_points = self.sa1(xyz, norm_plt)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        x = self.conv1(l1_points) 
        x = self.bn1(x)
        x = self.drop1 (x)
        x = F.relu(x)
        x = self.conv2 (x)
        x = F.relu(x)
        x = x.view(x.shape[0],16,32)
        x = x.permute(1,0,2)
        self.lstm.flatten_parameters()
        x, _ =self.lstm(x)
        x = x[x.shape[0]-1,:,:].view(1,x.shape[1],x.shape[2])
        x = x.permute(1,0,2)
        x = x.view(x.shape[0],1,16,16)

        x = self.ct1(x) 
        x = self.b1(x) 
        x = F.relu(x)
        x = self.ct2(x) 
        x = self.b2(x) 
        x = F.relu(x)
        x = self.ct3(x) 
        x = self.b3(x)
        x = F.relu(x)
        x = self.ct4(x) 
        x = self.b4(x) 
        x = F.relu(x)
        x = self.ct5(x) 
        x = self.b5(x) 
        x = F.relu(x)
        x = self.ct6(x) 
        x = self.b6(x) 
        x = F.relu(x)
        x = self.ct7(x) 
        x = self.b7(x) 
        x = F.relu(x)
        x = self.ct8(x) 
        x = self.b8(x) 
        x = F.relu(x)
        x = self.ct9(x) 
        x = self.b9(x) 
        x = F.relu(x)
        x = self.ct10(x) 
        x = self.b10 (x)
        x = F.relu(x)
        x = self.ct11(x) 
        x = self.b11 (x)
        x = F.relu(x)
        x = self.ct12(x) 
        x = self.b12(x) 
        x = F.relu(x)
        x = self.ct13(x) 
        x = self.b13(x) 
        x = F.relu(x)
        x = self.ct14(x) 
        x = self.b14(x) 
        x = F.relu(x)
        x = self.ct15(x) 
        x = self.b15(x) 
        x = F.relu(x)
        x = self.ct16(x) 
        x = self.b16(x) 
        x = F.relu(x)
        x = self.ct17(x) 
        x = self.b17 (x)
        x = F.relu(x)
        x = self.ct18(x) 
        x = self.b18(x) 
        x = F.relu(x)
        x = self.ct19(x) 
        x = self.b19(x)
        x = F.relu(x)
        x = self.ct20(x) 
        
        x = x.view(x.shape[0],x.shape[2],x.shape[3])


        return x

class PointNet2_modified(nn.Module):
    def __init__(self):
        super(PointNet2_modified, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.2, 0.4], [16, 32, 64], 0+3, [[16, 16, 32], [32, 32, 64], [32, 58, 64]])
        #self, npoint, radius_list, nsample_list, in_channel, mlp_list
        #self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])

        self.sa2 = PointNetSetAbstractionMsg(64, [0.4,0.8], [64, 128], 32+64+64, [[64, 64, 128], [128, 196, 128]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+128 + 3, mlp=[128, 256, 256], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=416, mlp=[256, 128])
        self.lstm = nn.LSTM(256,256)
        self.ct1 = ConvTranspose2d(1,8, (3,3))
        self.b1 = BatchNorm2d(8)

        self.ct2 = ConvTranspose2d(8,16, (3,3))
        self.b2 = BatchNorm2d(16)

        self.ct3 = ConvTranspose2d(16,32, (3,3))
        self.b3 = BatchNorm2d(32)
        self.ct4 = ConvTranspose2d(32,32, (3,3))
        self.b4 = BatchNorm2d(32)
        self.ct5 = ConvTranspose2d(32,32, (3,3))
        self.b5 = BatchNorm2d(32)
        self.ct6 = ConvTranspose2d(32,64, (3,3))
        self.b6 = BatchNorm2d(64)
        self.ct7 = ConvTranspose2d(64,64, (3,3))
        self.b7 = BatchNorm2d(64)
        self.ct8 = ConvTranspose2d(64,64, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.b8 = BatchNorm2d(64)
        self.ct9 = ConvTranspose2d(64,64, (3,3))
        self.b9 = BatchNorm2d(64)
        self.ct10 = ConvTranspose2d(64,128, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.b10 = BatchNorm2d(128)
        self.ct11 = ConvTranspose2d(128,128, (3,3))
        self.b11 = BatchNorm2d(128)
        self.ct12 = ConvTranspose2d(128,64, (3,3),stride=(4,2),padding=(0,1),output_padding=(1,1))
        self.b12 = BatchNorm2d(64)
        self.ct13 = ConvTranspose2d(64,32, (3,3))
        self.b13 = BatchNorm2d(32)
        self.ct14 = ConvTranspose2d(32,32, (3,3))
        self.b14 = BatchNorm2d(32)
        self.ct15 = ConvTranspose2d(32,32, (3,1))
        self.b15 = BatchNorm2d(32)
        self.ct16 = ConvTranspose2d(32,18, (3,1))
        self.b16 = BatchNorm2d(18)
        self.ct17 = ConvTranspose2d(18,18, (1,1))
        self.b17 = BatchNorm2d(18)
        self.ct18 = ConvTranspose2d(18,16, (1,1))
        self.b18= BatchNorm2d(16)
        self.ct19 = ConvTranspose2d(16,8, (1,1))
        self.b19 = BatchNorm2d(8)
        self.ct20 = ConvTranspose2d(8,1, (1,1))
    def forward(self, xyz, norm_plt):
        # Set Abstraction layers
       
        l1_xyz, l1_points = self.sa1(xyz, norm_plt)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
   
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        x =l1_points
        x = x.permute(1,0,2)
        #self.lstm.flatten_parameters()
        x, _ =self.lstm(x)
        x = x[x.shape[0]-1,:,:].view(1,x.shape[1],x.shape[2])
        x = x.permute(1,0,2)
        x = x.view(x.shape[0],1,16,16)

        x = self.ct1(x) 
        x = self.b1(x) 
        x = F.relu(x)
        x = self.ct2(x) 
        x = self.b2(x) 
        x = F.relu(x)
        x = self.ct3(x) 
        x = self.b3(x)
        x = F.relu(x)
        x = self.ct4(x) 
        x = self.b4(x) 
        x = F.relu(x)
        x = self.ct5(x) 
        x = self.b5(x) 
        x = F.relu(x)
        x = self.ct6(x) 
        x = self.b6(x) 
        x = F.relu(x)
        x = self.ct7(x) 
        x = self.b7(x) 
        x = F.relu(x)
        x = self.ct8(x) 
        x = self.b8(x) 
        x = F.relu(x)
        x = self.ct9(x) 
        x = self.b9(x) 
        x = F.relu(x)
        x = self.ct10(x) 
        x = self.b10 (x)
        x = F.relu(x)
        x = self.ct11(x) 
        x = self.b11 (x)
        x = F.relu(x)
        x = self.ct12(x) 
        x = self.b12(x) 
        x = F.relu(x)
        x = self.ct13(x) 
        x = self.b13(x) 
        x = F.relu(x)
        x = self.ct14(x) 
        x = self.b14(x) 
        x = F.relu(x)
        x = self.ct15(x) 
        x = self.b15(x) 
        x = F.relu(x)
        x = self.ct16(x) 
        x = self.b16(x) 
        x = F.relu(x)
        x = self.ct17(x) 
        x = self.b17 (x)
        x = F.relu(x)
        x = self.ct18(x) 
        x = self.b18(x) 
        x = F.relu(x)
        x = self.ct19(x) 
        x = self.b19(x)
        x = F.relu(x)
        x = self.ct20(x) 
        
        x = x.view(x.shape[0],x.shape[2],x.shape[3])


        return x


class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz,points):
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x
class PointNet2_conv2d_lstm(nn.Module):
    def __init__(self):
        super(PointNet2_conv2d_lstm, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 0+3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)
        self.convlstm = ConvLSTM2D(input_channels=16, hidden_channels=[32,8], kernel_size=3, step=5,
                        effective_step=[4]).cuda()
        #self.conv3  = nn.Conv2d(8,1,(1,1))
        self.ct1 = ConvTranspose2d(1,8, (3,3))
        self.b1 = BatchNorm2d(8)

        self.ct2 = ConvTranspose2d(8,16, (3,3))
        self.b2 = BatchNorm2d(16)

        self.ct3 = ConvTranspose2d(16,32, (3,3))
        self.b3 = BatchNorm2d(32)
        self.ct4 = ConvTranspose2d(32,32, (3,3))
        self.b4 = BatchNorm2d(32)
        self.ct5 = ConvTranspose2d(32,32, (3,3))
        self.b5 = BatchNorm2d(32)
        self.ct6 = ConvTranspose2d(32,64, (3,3))
        self.b6 = BatchNorm2d(64)
        self.ct7 = ConvTranspose2d(64,64, (3,3))
        self.b7 = BatchNorm2d(64)
        self.ct8 = ConvTranspose2d(64,64, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.b8 = BatchNorm2d(64)
        self.ct9 = ConvTranspose2d(64,64, (3,3))
        self.b9 = BatchNorm2d(64)
        self.ct10 = ConvTranspose2d(64,128, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.b10 = BatchNorm2d(128)
        self.ct11 = ConvTranspose2d(128,128, (3,3))
        self.b11 = BatchNorm2d(128)
        self.ct12 = ConvTranspose2d(128,64, (3,3),stride=(4,2),padding=(0,1),output_padding=(1,1))
        self.b12 = BatchNorm2d(64)
        self.ct13 = ConvTranspose2d(64,32, (3,3))
        self.b13 = BatchNorm2d(32)
        self.ct14 = ConvTranspose2d(32,32, (3,3))
        self.b14 = BatchNorm2d(32)
        self.ct15 = ConvTranspose2d(32,32, (3,1))
        self.b15 = BatchNorm2d(32)
        self.ct16 = ConvTranspose2d(32,18, (3,1))
        self.b16 = BatchNorm2d(18)
        self.ct17 = ConvTranspose2d(18,18, (1,1))
        self.b17 = BatchNorm2d(18)
        self.ct18 = ConvTranspose2d(18,16, (1,1))
        self.b18= BatchNorm2d(16)
        self.ct19 = ConvTranspose2d(16,8, (1,1))
        self.b19 = BatchNorm2d(8)
        self.ct20 = ConvTranspose2d(8,1, (1,1))


    def forward(self, xyz, norm_plt):
        # Set Abstraction layers
       
        l1_xyz, l1_points = self.sa1(xyz, norm_plt)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        x = self.conv1(l1_points) 
        x = self.bn1(x)
        x = self.drop1 (x)
        x = F.relu(x)
        x = self.conv2 (x)
        x = F.relu(x)
        x = x.view(x.shape[0],16,8,4)
        #x = x.permute(1,0,2,3)
        output, _ =self.convlstm(x)
        #x = x[x.shape[0]-1,:,:].view(1,x.shape[1],x.shape[2]*x.shape[3])
        #x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
        x = output[len(output)-1]
        
        #x = x.permute(1,0,2)
        #x = self.conv3(x)
        x = x.view(x.shape[0],1,16,16)

        x = self.ct1(x) 
        x = self.b1(x) 
        x = F.relu(x)
        x = self.ct2(x) 
        x = self.b2(x) 
        x = F.relu(x)
        x = self.ct3(x) 
        x = self.b3(x)
        x = F.relu(x)
        x = self.ct4(x) 
        x = self.b4(x) 
        x = F.relu(x)
        x = self.ct5(x) 
        x = self.b5(x) 
        x = F.relu(x)
        x = self.ct6(x) 
        x = self.b6(x) 
        x = F.relu(x)
        x = self.ct7(x) 
        x = self.b7(x) 
        x = F.relu(x)
        x = self.ct8(x) 
        x = self.b8(x) 
        x = F.relu(x)
        x = self.ct9(x) 
        x = self.b9(x) 
        x = F.relu(x)
        x = self.ct10(x) 
        x = self.b10 (x)
        x = F.relu(x)
        x = self.ct11(x) 
        x = self.b11 (x)
        x = F.relu(x)
        x = self.ct12(x) 
        x = self.b12(x) 
        x = F.relu(x)
        x = self.ct13(x) 
        x = self.b13(x) 
        x = F.relu(x)
        x = self.ct14(x) 
        x = self.b14(x) 
        x = F.relu(x)
        x = self.ct15(x) 
        x = self.b15(x) 
        x = F.relu(x)
        x = self.ct16(x) 
        x = self.b16(x) 
        x = F.relu(x)
        x = self.ct17(x) 
        x = self.b17 (x)
        x = F.relu(x)
        x = self.ct18(x) 
        x = self.b18(x) 
        x = F.relu(x)
        x = self.ct19(x) 
        x = self.b19(x)
        x = F.relu(x)
        x = self.ct20(x) 




        return x

    #for some reason writing the decoder this way in all 
    #of these architecutres an error about type 
    #missmatch:


    """
        #__iniit__
        self.decoder.append(ConvTranspose2d(1,8, (3,3)))
        self.decoder.append( BatchNorm2d(8))
        #self.decoder.append(ReLU())
        self.decoder.append(ConvTranspose2d(8,16, (3,3)))
        self.decoder.append( BatchNorm2d(16))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(16,32, (3,3)))
        self.decoder.append( BatchNorm2d(32))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(32,32, (3,3)))
        self.decoder.append(BatchNorm2d(32))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(32,32, (3,3)))
        self.decoder.append(BatchNorm2d(32))
        #self.decoder.append(ReLU())
        self.decoder.append(ConvTranspose2d(32,64, (3,3)))
        self.decoder.append(BatchNorm2d(64))
        #self.decoder.append(ReLU())
        self.decoder.append(ConvTranspose2d(64,64, (3,3)))
        self.decoder.append(BatchNorm2d(64))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(64,64, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1)))
        self.decoder.append( BatchNorm2d(64))
        #self.decoder.append(ReLU())
        self.decoder.append(ConvTranspose2d(64,64, (3,3)))
        self.decoder.append( BatchNorm2d(64))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(64,64, (3,3),stride=(2,2),padding=(1,1),output_padding=(1,1)))
        self.decoder.append(BatchNorm2d(64))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(64,128, (3,3)))
        self.decoder.append(BatchNorm2d(128))
        #self.decoder.append(ReLU())
        self.decoder.append(ConvTranspose2d(128,64, (3,3),stride=(4,2),padding=(0,1),output_padding=(1,1)))
        self.decoder.append( BatchNorm2d(64))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(64,32, (3,3)))
        self.decoder.append( BatchNorm2d(32))
        #self.decoder.append(ReLU())
        self.decoder.append(ConvTranspose2d(32,32, (3,3)))
        self.decoder.append( BatchNorm2d(32))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(32,32, (3,1)))
        self.decoder.append( BatchNorm2d(32))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(32,18, (3,1)))
        self.decoder.append( BatchNorm2d(18))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(18,18, (1,1)))
        self.decoder.append(BatchNorm2d(18))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(18,16, (1,1)))
        self.decoder.append( BatchNorm2d(16))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(16,8, (1,1)))
        self.decoder.append( BatchNorm2d(8))
        #self.decoder.append(ReLU())
        self.decoder.append( ConvTranspose2d(8,1, (1,1)))


        #__forward__
        for ix in  range(0,len(self.decoder)-1,2):
            x = self.decoder[ix](x)
            x = self.decoder[ix+1](x)
            x = F.relu(x)
        x = self.decoder[len(self.decoder)-1](x)
 
        x = x.view(x.shape[0],x.shape[2],x.shape[3])




    """


