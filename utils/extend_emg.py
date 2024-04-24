import cupy as np
from cupy.linalg import eigh
from typing import Tuple


def extend_EMG(EMGInput: np.ndarray, R: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extends the EMG input matrix by creating shifted versions of the input signal
    and then performs mean centering and whitening on the extended matrix.

    Args:
    - EMGInput (np.ndarray): Input EMG matrix (N x NumCh1).
    - R (int): Number of shifts to create.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the whitened EMG matrix and the whitening matrix.
    """
    if EMGInput.size == 0:
        raise ValueError("Input EMG matrix is empty.")
    if R < 0:
        raise ValueError("Number of shifts R must be non-negative.")

    N, NumCh1 = EMGInput.shape

    # Create an extended matrix with shifted columns
    EMGExtended = np.zeros((N, NumCh1 * (R + 1)))
    for i in range(R + 1):
        EMGExtended[max(0, i) : N, i * NumCh1 : (i + 1) * NumCh1] = EMGInput[0 : N - i]

    # Mean centering
    mean_vector = np.mean(EMGExtended, axis=0)
    EMGExtended -= mean_vector

    # Covariance matrix computation
    Rxx = np.dot(EMGExtended.T, EMGExtended) / (N - 1)

    # Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = eigh(Rxx)
    fudge = 1e-10  # Small constant to prevent division by zero
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + fudge))

    # Whitening transformation
    W = np.dot(eigenvectors, D_inv_sqrt).dot(eigenvectors.T)

    # Apply whitening matrix
    EMGOutput = np.dot(W, EMGExtended.T)

    return EMGOutput.T, W
