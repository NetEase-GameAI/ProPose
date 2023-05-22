import numpy as np


class DataLogger(object):
    """ Average data logger. """

    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def cal_mpjpe(pred, target, weight):
    """ Calculate MPJPE. 
    
    Args:
        pred (np.array): Predictions with shape `(B, J, 3)`.
        target (np.array): Ground-truth with shape `(B, J, 3)`.
        weight (np.array): Visibility with shape `(B, J, 3)`

    Returns:
        float: MPJPE.
        int: The number of considered people in batch.
    """
    assert pred.ndim == 3
    cnt = 0
    mpjpe = 0
    for b in range(pred.shape[0]):
        w_3d = weight[b, :, 2:3] # (J, 1)
        w_sum = np.sum(w_3d)
        if w_sum < 1:  # all keypoints are not annotated
            continue
        dist = (pred[b] - target[b]) * w_3d  # (J, 3)
        dist = np.linalg.norm(dist, axis=-1) # (J,)
        mpjpe += np.sum(dist) / w_sum
        cnt += 1
    
    if cnt == 0:
        return -1, 0
    else:
        return mpjpe/cnt, cnt


def reconstruction_error(S1, S2):
    """ Do Procrustes alignment and compute reconstruction error. """
    S1_hat = compute_similarity_transform_batch(S1, S2)
    return S1_hat


def compute_similarity_transform(S1, S2):
    """ MPJPE-PA.
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R (3, 3) rotation matrix, t (3, 1) translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1]), (S1.shape, S2.shape)

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """ Batched version of compute_similarity_transform. """
    if S1.ndim == 2:
        S1_hat = compute_similarity_transform(S1.copy(), S2.copy())
    else:
        S1_hat = np.zeros_like(S1)
        for i in range(S1.shape[0]):
            S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat