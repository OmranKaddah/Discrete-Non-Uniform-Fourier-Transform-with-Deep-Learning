import tensorflow as tf
import numpy as np
from custom_layers.pointnet2_utils import index_points, farthest_point_sample, query_ball_point
from tensorflow.python.keras.layers import Conv2D, BatchNormalization
from tensorflow.python.keras.activations import relu
class PointNetSetAbstractionMsg(tf.keras.layers.Layer):
    def __init__(self, npoint, radius_list, nsample_list, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = []
        self.bn_blocks = []
        for i in range(len(mlp_list)):
            convs = []
            bns = []
            #last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(Conv2D(out_channel,1))
                convs.append(Conv2D(out_channel,1))
                bns.append(BatchNormalization())
                #last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def call(self, x):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz , points = x
        xyz = tf.transpose(xyz,(0,2,1))
        #xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = tf.transpose(points,(0,2,1))
            #points = points.permute(0, 2, 1)

        b, n, c = xyz.shape
        B = b.value
        N = n.value
        C = c.value
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz = tf.subtract(grouped_xyz , tf.reshape(new_xyz,(B,S,1,C)))
            #grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)

                grouped_points = tf.concat([grouped_points, grouped_xyz],-1)
                #grouped_points = np.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = tf.transpose(grouped_points,(0,3,2,1))
            #grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  relu(bn(conv(grouped_points)))
            new_points = tf.reduce_max(grouped_points,2)
            #new_points = np.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = tf.transpose(new_xyz,(0,2,1))
        #new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = tf.concat(new_points_list,-1)
        return (new_xyz, new_points_concat)