import torch
import torch.nn as nn
from umeyama import umeyama
import time


class ICPMatcher(nn.Module):

    def __init__(self, max_nearest_distance=0.2, max_iteration=100):
        self.threshold = max_nearest_distance ** 2
        self.max_iteration = max_iteration

    def forward(self, source, target):
        assert source.ndim == 2 and target.ndim == 2
        assert source.size(1) == target.size(1)
        return self.icp_match(source, target)

    def icp_match(self, source, target):
        iter_num = 0
        last_corresponces = torch.tensor(0.)
        total_transf_mat = torch.eye(4).type_as(source)
        while iter_num < self.max_iteration:
            transf_mat, corresponces = self.icp_step(source, target)
            source = self.transform(source, transf_mat)
            # target = self.transform(target, transf_mat)
            iter_num += 1
            total_transf_mat = transf_mat @ total_transf_mat

            if corresponces.size() == last_corresponces.size() and \
                    torch.allclose(corresponces, last_corresponces):
                break
            else:
                last_corresponces = corresponces

        return total_transf_mat

    def icp_step(self, source, target):
        # t0 = time.time()
        min_distances, corresponce_indices = self.find_nearest_neighbor(source, target)
        mask = (min_distances < self.threshold)
        
        filtered_source = source[mask]
        if filtered_source.size(0) < 4:
            raise ValueError('Too few inliers for icp')

        corresponce_indices = corresponce_indices[mask]
        corresponces = torch.index_select(target, dim=0, index=corresponce_indices)

        # t1 = time.time()
        transf_mat = umeyama(filtered_source.cpu(), corresponces.cpu(), False)
        transf_mat = transf_mat.type_as(source)
        # t2 = time.time()
        # print(t1 - t0, t2 - t1)
        return transf_mat, corresponces

    @staticmethod
    def transform(points, transf_mat):
        ones = torch.ones(points.size(0), 1).type_as(points)
        points = torch.cat([points, ones], dim=1)  # [N, 4]
        points = points @ transf_mat.T

        return points[:, :3]

    @ staticmethod
    def find_nearest_neighbor(source, target):

        N1, D1 = source.size()
        N2, D2 = target.size()

        source = source.unsqueeze(1).expand(-1, N2, -1)  # [N1, N2, D]
        target = target.unsqueeze(0).expand(N1, -1, -1)  # [N1, N2, D]
        distances = torch.sum((source - target) ** 2, dim=2)  # [N1, N2]

        min_distances, min_indices = torch.min(distances, dim=1)  # [N1]
        return min_distances, min_indices

