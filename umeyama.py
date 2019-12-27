# # License (Modified BSD) # Copyright (C) 2011, the scikit-image team All rights reserved. # # Redistribution and
# use in source and binary forms, with or without modification, are permitted provided that the following conditions
# are met: # # Redistributions of source code must retain the above copyright notice, this list of conditions and the
#  following disclaimer. # Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  Neither the name of skimage nor the names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission. # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# umeyama function from scikit-image/skimage/transform/_geometric.py

import numpy as np
import torch


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.size(0)
    dim = src.size(1)

    # Compute mean of src and dst.
    src_mean = src.mean(dim=0)  # [N]
    dst_mean = dst.mean(dim=0)  # [N]

    # Subtract mean from src and dst.
    src_demean = src - src_mean  # [M, N]
    dst_demean = dst - dst_mean  # [M, N]

    # Eq. (38).
    A = (dst_demean.transpose(0, 1) @ src_demean) / num  # [M, N]

    # Eq. (39).
    d = torch.ones(dim).type_as(src)  # [N]
    if torch.det(A) < 0:
        d[dim - 1] = -1

    T = torch.eye(dim + 1).type_as(src)

    U, S, VH = torch.svd(A)
    V = VH.transpose(0, 1)  # difference between numpy and pytorch

    # Eq. (40) and (43).
    rank = torch.matrix_rank(A)
    if rank == 0:
        return float('nan') * T
    elif rank == dim - 1:
        if torch.det(U) * torch.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ torch.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ torch.diag(d) @ V.transpose(0, 1)

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(dim=0).sum() * (S @ d)
    else:
        scale = 1.0

    src_mean = src_mean.unsqueeze(1)  # [N, 1]
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean).squeeze(1)
    T[:dim, :dim] *= scale

    return T
