
import tensorflow as tf
import numpy



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    print(src.shape)
    B, N, _ = src.shape
    #B = b.value
    #N = n.value
    M= dst.shape[1]
    #M = m.value
    dist =  tf.matmul(src,tf.transpose(dst,perm=(0,2,1))) * -2
            
    dist  = tf.math.add(dist, tf.reshape(tf.reduce_sum(tf.square(src),-1),(B,N,1)))

    dist  = tf.math.add(dist, tf.reshape(tf.reduce_sum(tf.square(dst),-1),(B,1,M)))

    return dist
    
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B,N,C = points.shape

    _,S = idx.get_shape()
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = tf.tile(tf.reshape(tf.range(0,B,dtype=tf.int64),view_shape),repeat_shape)
    #batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    indicies = tf.concat([tf.reshape(idx,(tf.size(idx),1)), tf.reshape(batch_indices,(tf.size(batch_indices),1))],1)
    indicies = tf.reshape(indicies,(B,S,C))
    new_points = tf.gather_nd(points,indicies)
    #new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    
    B, N, C = xyz.get_shape().as_list()
    
    #B = 1
    #N = n.value
    #distance = tf.math.multiply(tf.ones((B,N)) ,1e10)
    distance = tf.ones((B,N)) 
    farthest = tf.random.uniform((B,1),0,N,dtype=tf.int64)
    batch_indices = tf.range(0,B,dtype= tf.int64)
    
    unpacked = []
    #const = tf.constant([1])
    for i in range(npoint):
        
        farthest = tf.reshape(farthest,(B,1))
        unpacked.append(farthest)

        con = tf.concat((tf.reshape(batch_indices,(B,1)),farthest),axis=1)
        centroid = tf.reshape(tf.gather_nd(xyz,con),(B,1,C))
        dist = tf.math.reduce_sum(tf.math.square(tf.subtract(xyz,centroid)),-1)
        
        

        mask = tf.greater(distance,dist)
        update = tf.boolean_mask(dist,mask)
        ind = tf.where(mask)
        distance = tf.tensor_scatter_nd_update(distance,ind,update)
        farthest = tf.math.argmax(distance,-1)
        
    centroids = tf.concat(unpacked,1)
    centroids = tf.reshape(centroids,(B, npoint))

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B,N,C = xyz.get_shape().as_list()
    
    #B = b.value
    #N = n.value
    #C = c.value
    S = new_xyz.get_shape().as_list()[1]

    group_idx = tf.tile((tf.reshape(tf.range(0,N),(1,1,N))) ,[B, S, 1])

    sqrdists = square_distance(new_xyz, xyz)
    mask = tf.greater(sqrdists,tf.square(radius))

    indecies = tf.where(mask)
    Ns = tf.ones(tf.shape(group_idx)) * N
    Ns = tf.boolean_mask(Ns,mask)
    Ns = tf.dtypes.cast(Ns,tf.int32)
    un_group_idx = tf.tensor_scatter_nd_update(group_idx,indecies,Ns)

    #group_idx[sqrdists > radius ** 2] = N
    group_idx  = tf.sort(un_group_idx)[:,:,:nsample]

    #group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    temp = group_idx[:, :, 0]

    group_first = tf.tile(tf.reshape(temp,(B,S,1)),[1,1,nsample])
    #group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = tf.math.equal(group_idx, 1)
    #mask = tf.equal(group_idx, tf.ones(group_idx.shape))
    #mask = group_idx == N
    updates = tf.boolean_mask(group_first,mask)
    indecies = tf.where_v2(updates)
    print("##############################################")
    print(updates.shape)
    print(indecies.shape)
    group_idx  = tf.tensor_scatter_nd_update(group_idx,indecies,updates)
    #group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, _, C = xyz.get_shape().as_list()
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = tf.subtract(grouped_xyz,tf.reshape(new_xyz,(B,S,1,C)))
    #grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = tf.concat([grouped_xyz_norm,grouped_points],-1)
        #new_points = np.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
        