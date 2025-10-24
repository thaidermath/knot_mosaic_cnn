"""
utils_npy.py

Small utility for reading .npy tile-prob files used by the project.

Example:
    from utils_npy import load_tile_probs
    arr = load_tile_probs('tile_probs_from_matrices/9_9/tile_probs.npy')
    print(arr.shape)  # e.g. (11, 6, 6)

"""
from pathlib import Path
import numpy as np


def load_tile_probs(path):
    """Load a NumPy .npy file and return the array.

    Args:
        path (str or Path): Path to .npy file.

    Returns:
        np.ndarray: Loaded array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    arr = np.load(str(p))
    return arr


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        a = load_tile_probs(sys.argv[1])
        print('shape:', a.shape)
    else:
        print('Usage: python utils_npy.py path/to/tile_probs.npy')
