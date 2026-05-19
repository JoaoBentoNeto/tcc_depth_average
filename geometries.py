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
        (lattice_length, 2 + round((height / resolution) / aspect_ratio)),
        dtype=np.float64,
    )

    duct_2d[:, 0] = 0
    duct_2d[:, -1] = 0

    return duct_2d


def create_parallel_plates(
    lattice_width: int, lattice_length: int
) -> np.ndarray:
    """Generates a 2D domain map for parallel plates.

    Creates a boolean-like integer array where 1 represents the fluid and
    0 represents the solid boundaries (walls) on the lateral edges.

    Args:
        lattice_width: The width in voxels.
        lattice_length: The length in the flow direction in voxels.

    Returns:
        A 2D NumPy array representing the parallel plates geometry.
    """
    parallel_plates = np.ones((lattice_length, lattice_width), dtype=np.float64)

    return parallel_plates


def create_unit_cell(
    radius: float,
    relative_distance: float,
    vertical_repeats: int = 1,
    horizontal_repeats: int = 1,
) -> np.ndarray:
    """Generates a 2D map repeating a unit cell with 5 cylinders.

    Args:
        radius: The radius of the obstacles in pixels.
        relative_distance: Distance between cylinders as a function of radius.
        vertical_repeats: Number of times the cell is repeated vertically.
        horizontal_repeats: Number of times the cell is repeated horizontally.

    Returns:
        A 2D NumPy array of type float where 0 is solid and 1 is fluid.
    """
    cell_size = (2.0 + relative_distance) * radius

    center_min = 0.0
    center_mid = cell_size / 2.0
    center_max = cell_size

    total_width = int(np.round(horizontal_repeats * cell_size))
    total_height = int(np.round(vertical_repeats * cell_size))

    x = np.arange(total_width)
    y = np.arange(total_height)
    xx, yy = np.meshgrid(x, y)

    x_local = xx % cell_size
    y_local = yy % cell_size

    r2 = radius**2

    center_mask = (x_local - center_mid) ** 2 + (
        y_local - center_mid
    ) ** 2 <= r2

    top_left_mask = (x_local - center_min) ** 2 + (
        y_local - center_min
    ) ** 2 <= r2
    top_right_mask = (x_local - center_max) ** 2 + (
        y_local - center_min
    ) ** 2 <= r2
    bot_left_mask = (x_local - center_min) ** 2 + (
        y_local - center_max
    ) ** 2 <= r2
    bot_right_mask = (x_local - center_max) ** 2 + (
        y_local - center_max
    ) ** 2 <= r2

    solid = (
        center_mask
        | top_left_mask
        | top_right_mask
        | bot_left_mask
        | bot_right_mask
    )

    domain = (~solid).astype(np.uint8)

    return domain.astype(np.float64)
