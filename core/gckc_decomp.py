import cupy as np
from tqdm import trange


def gCKC_decomposition(sig_mat, MU_num, max_iterations_gradient, lr, init_indices=None):
    num_samples, num_channels = sig_mat.shape

    # Compute the correlation matrix of the measurements
    C_xx = np.cov(sig_mat, rowvar=False)
    C_xx_inv = np.linalg.inv(C_xx)

    # Calculate the activity index IA(n)
    IA = np.einsum("ij,ji->i", np.dot(sig_mat, C_xx_inv), sig_mat.T)
    n1_indices = np.argsort(-IA)[:MU_num] if init_indices is None else init_indices

    # Initialize c_tjx using x(n1) and normalize
    c_tjx = np.zeros((num_channels, MU_num))
    for j in range(MU_num):
        c_tjx[:, j] = sig_mat[n1_indices[j], :]
        norm = np.linalg.norm(c_tjx[:, j])
        if norm > 0:
            c_tjx[:, j] /= norm

    # Placeholder for estimated pulse trains
    estimated_pulse_trains = np.zeros((num_samples, MU_num))

    for _ in trange(max_iterations_gradient):
        for j in range(MU_num):
            # Update estimated pulse train using current cross-correlation vector
            estimated_pulse_trains[:, j] = sig_mat @ C_xx_inv @ c_tjx[:, j]

            # Compute gradient based on a simple quadratic cost function derivative
            grad = sig_mat.T @ (2 * estimated_pulse_trains[:, j])

            # Update the cross-correlation vector using the gradient
            c_tjx[:, j] -= lr * grad
            norm = np.linalg.norm(c_tjx[:, j])
            if norm > 0:  # Normalize to prevent numerical instability
                c_tjx[:, j] /= norm

    return estimated_pulse_trains
