# Pytorch ICP

Iterative Closest Point (ICP) algorithm implemented by PyTorch.

## Requirements

```
pytorch
numpy
open3d (optional, it is only used in test.py and not neccesary)
```

## How to use

``` python

# You can also load point cloud without open3d
source_pcd = o3d.io.read_point_cloud('data/source.pcd')
target_pcd = o3d.io.read_point_cloud('data/target.pcd')
target_pcd.translate(np.array([0.1, 0.2, 0]))
source = np.asarray(source_pcd.points)
target = np.asarray(target_pcd.points)
source = torch.from_numpy(source.copy())  #.cuda()
target = torch.from_numpy(target.copy())  #.cuda()

# Assume sourse and target cloud point is already loaded into pytorch tensor
# [n_points, n_dim]
matcher = ICPMatcher(max_nearest_distance=0.2, max_iteration=1000)
transf_mat = matcher.icp_match(source, target)

```

## Performance

Run `test.py`.

Comparision between Open3d ICP and Pytorch ICP (in CPU).
```
Open3d: 8.227141857147217s for 100 loops
Transformation:
[[ 0.99953283 -0.03056346  0.         -0.02351535]
 [ 0.03056346  0.99953283  0.          0.19804616]
 [ 0.          0.          1.         -0.13614899]
 [ 0.          0.          0.          1.        ]]

Pytorch ICP: 3.825660228729248s for 100 loops
Transformation:
tensor([[ 0.9995, -0.0306,  0.0000, -0.0235],
        [ 0.0306,  0.9995,  0.0000,  0.1980],
        [ 0.0000,  0.0000,  1.0000, -0.1361],
        [ 0.0000,  0.0000,  0.0000,  1.0000]], dtype=torch.float64)
```

Note: The performace of Pytorch ICP in GPU is not as good as CPU now.
