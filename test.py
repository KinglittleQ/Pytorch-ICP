import open3d as o3d
import torch
import copy
import numpy as np
from icp_matcher import ICPMatcher
from time import time

def test():
    matcher = ICPMatcher(max_iteration=1000)

    source_pcd = o3d.io.read_point_cloud('data/source.pcd')
    target_pcd = o3d.io.read_point_cloud('data/target.pcd')
    target_pcd.translate(np.array([0.1, 0.2, 0]))

    t0 = time()
    for i in range(100):
        reg = open3d_icp_match(source_pcd, target_pcd, max_iteration=1000)
    print('Open3d: {}s for 100 matches'.format(time() - t0))
    print(reg.transformation)

    source = np.asarray(source_pcd.points)
    target = np.asarray(target_pcd.points)
    source = torch.from_numpy(source.copy())#.cuda()
    target = torch.from_numpy(target.copy())#.cuda()

    t0 = time()
    for i in range(100):
        transf_mat = matcher.icp_match(source, target)
    print('Pytorch ICP: {}s for 100 matches'.format(time() - t0))
    print(transf_mat)


def open3d_icp_match(source_pcd,
                     target_pcd,
                     threshold=0.2,
                     max_iteration=100,
                     method='point2point'):

    method = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = np.eye(4)
    reg_p2p = o3d.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=threshold,
        init=trans_init,
        criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
        estimation_method=method
    )

    return reg_p2p


if __name__ == '__main__':
    test()
