import numpy as np


def create_duct_2d(
    aspect_ratio: float, height: float, resolution: float, lattice_length: int
) -> np.ndarray:
    """Generates a 2D domain map for a rectangular duct.

    Creates a boolean-like integer array where 1 represents the fluid and
    0 represents the solid boundaries (walls) on the lateral edges.

    Args:
        aspect_ratio: The ratio of height to width (AR = h / w).
        height: The height of the channel in micrometers.
        resolution: Conversion factor in micrometers/voxel.
        lattice_length: The length of the duct in the flow direction in voxels.

    Returns:
        A 2D NumPy array representing the duct geometry.
    """
    duct_2d = np.ones(
        (lattice_length, 2 + round((height / resolution) / aspect_ratio))
    )

    duct_2d[:, 0] = 0
    duct_2d[:, -1] = 0

    return duct_2d
