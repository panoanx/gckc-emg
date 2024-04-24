import cupy as np


def load_mat(mat_path: str) -> np.ndarray:
    """
    Load a .mat file.

    Args:
    - mat_path (str): Path to the .mat file.

    Returns:
    - np.ndarray: Data from the .mat file.
    """

    try:
        import h5py

        f = h5py.File(mat_path, "r")
        data = f.get("data")
        data = np.array(data)
    except OSError:
        import scipy.io

        mat = scipy.io.loadmat(mat_path)
        data = mat["data"]

    return np.array(data)
