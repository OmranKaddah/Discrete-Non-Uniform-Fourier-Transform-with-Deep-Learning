import tensorflow as tf
import numpy as np

from custom_layers.pointnet2_utils import index_points, farthest_point_sample, query_ball_point, square_distance
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Conv1D
from tensorflow.python.keras.activations import relu

class PointNetFeaturePropagation(tf.keras.layers.Layer):
    def __init__(self, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = []
        self.mlp_bns = []
        for out_channel in mlp:
            self.mlp_convs.append(Conv1D( out_channel, 1))
            self.mlp_bns.append(BatchNormalization(out_channel))
            

    def call(self, x):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1, xyz2, points1, points2 = x
        xyz1 = tf.transpose(xyz1,(0,2,1))
        xyz2 = tf.transpose(xyz2,(0,2,1))
        
        points2 = tf.transpose(points2,(0,2,1))

        b, n, c = xyz1.shape
        _, s, _ = xyz2.shape
        B = b.value
        N = n.value
        C = c.value
        S = s.value
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            #dists, idx = dists.sort(dim=-1)
            dists = tf.sort(dists)
            idx = tf.math.argmax(dists,-1)
            dists = dists[:, :, :3],
            idx =  idx[:, :, :3]  # [B, N, 3]
            mask = tf.math.greater(dists,1e-10)
            update = tf.ones(tf.shape(dists)) * 1e-10
            update = tf.boolean_mask(update,mask)
            dists = tf.tensor_scatter_nd_update(dists,tf.where(mask),update)
            #dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = tf.math.divide(weight,tf.reshape(tf.reduce_sum(weight,-1),(B,N,3))) # [B, N, 3]
            interpolated_points = tf.math.reduce_sum(tf.multiply(index_points(points2, idx),tf.reshape(weight,(B,N,3,1))),2)
        if points1 is not None:
            points1 = tf.transpose(points1,(0,2,1))
            new_points = tf.concat([points1, interpolated_points], -1)
        else:
            new_points = interpolated_points

        new_points = tf.transpose(new_points,(0,2,1))
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  relu(bn(conv(new_points)))
        return new_points