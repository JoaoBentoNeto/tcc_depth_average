from typing import Sequence

import numpy as np
import porespy as ps
from scipy.ndimage import distance_transform_edt, gaussian_filter


def create_duct(
    aspect_ratio: float, height: float, resolution: float, lattice_length: int
) -> np.ndarray:
    """Creates a 2D domain map for a rectangular duct.

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

    semi_width_map = ps.filters.local_thickness(duct_2d != 0)
    print(
        f"Rectangular duct maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels = {round(np.max(semi_width_map[semi_width_map > 0]) * 2) * resolution}  \u03bcm"
    )
    print(
        f"Rectangular duct minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels = {round(np.min(semi_width_map[semi_width_map > 0]) * 2) * resolution}  \u03bcm"
    )
    print(
        f"Rectangular duct mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2) * resolution}  \u03bcm\n"
    )
    return duct_2d


def create_parallel_plates(lattice_width: int, lattice_length: int) -> np.ndarray:
    """Creates a 2D domain map for parallel plates.

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
    """Creates a 2D map repeating a unit cell with 5 cylinders.

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

    center_mask = (x_local - center_mid) ** 2 + (y_local - center_mid) ** 2 <= r2

    top_left_mask = (x_local - center_min) ** 2 + (y_local - center_min) ** 2 <= r2
    top_right_mask = (x_local - center_max) ** 2 + (y_local - center_min) ** 2 <= r2
    bot_left_mask = (x_local - center_min) ** 2 + (y_local - center_max) ** 2 <= r2
    bot_right_mask = (x_local - center_max) ** 2 + (y_local - center_max) ** 2 <= r2

    solid = (
        center_mask | top_left_mask | top_right_mask | bot_left_mask | bot_right_mask
    )

    domain = (~solid).astype(np.uint8)

    semi_width_map = ps.filters.local_thickness(domain != 0)
    print(
        f"Unit cell maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Unit cell minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Unit cell mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return domain.astype(np.float64)


def create_constricted_channel(
    characteristic_length: float, resolution: float
) -> np.ndarray:
    """Creates a binary image of a 2D constricted channel with a slanted wall.

    This geometry serves as a non-trivial benchmark domain for fluid flow
    simulations, featuring a sudden expansion, a narrow throat constriction,
    and a gradually sloped divergent section.

    Args:
        characteristic_length: The physical characteristic length scale of the
          channel features.
        resolution: The physical size of each grid cell (dx).

    Returns:
        A 2D NumPy array of shape (nx, ny) where 1.0 represents the fluid
        domain and 0.0 represents the solid boundaries (walls).
    """
    l_nodes = round(2 * characteristic_length / resolution)

    nx = round(12 * l_nodes)
    ny = round(3.5 * l_nodes)

    image = np.zeros((nx, ny), dtype=np.float64)

    image[:l_nodes, 1:l_nodes] = 1.0
    image[l_nodes : 3 * l_nodes, 1:-1] = 1.0
    image[3 * l_nodes : 5 * l_nodes, 3 * l_nodes : -1] = 1.0

    for i in range(3 * l_nodes):
        image[5 * l_nodes + i, 3 * l_nodes - i : -1] = 1.0

    image[8 * l_nodes : 11 * l_nodes, 1:-1] = 1.0
    image[11 * l_nodes :, 1:l_nodes] = 1.0

    semi_width_map = ps.filters.local_thickness(image != 0)
    print(
        f"Constricted channel maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Constricted channel minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Constricted channel mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return image


def create_double_pore(
    eccentricity: Sequence[int],
    pore_rx: int,
    pore_ry: int,
    obstacle_rx: int,
    obstacle_ry: int,
) -> np.ndarray:
    """Creates a binary image of a double pore geometry with a smoothed channel.

    This function creates a domain representing a large pore connected to a
    straight channel, applies a Gaussian filter to smooth out the sharp corners
    at the channel-pore junctions, and places an eccentric solid obstacle
    inside the pore.

    Args:
        eccentricity: A sequence of two integers representing the [y, x] offset
          displacements for the center of the solid obstacle.
        pore_rx: The radius of the outer large pore along the x-axis.
        pore_ry: The radius of the outer large pore along the y-axis.
        obstacle_rx: The radius of the inner solid obstacle along the x-axis.
        obstacle_ry: The radius of the inner solid obstacle along the y-axis.

    Returns:
        A 2D NumPy array of shape (nx, ny) with dtype np.uint8, where 1
        represents the fluid domain (pore/channel) and 0 represents the solid
        domain (walls/obstacle).
    """
    nx = pore_rx * 3
    ny = int(round(pore_ry * 3))

    data = np.zeros((nx, ny), dtype=np.float64)
    x, y = np.ogrid[:nx, :ny]

    channel_width = pore_ry

    cx_pore, cy_pore = nx // 2, ny // 2
    cx_obstacle = nx // 2 + eccentricity[1]
    cy_obstacle = ny // 2 + eccentricity[0]

    smoothing_radius = channel_width / 4

    mask_channel = (y >= (ny // 2 - channel_width // 2)) & (
        y <= (ny // 2 + channel_width // 2)
    )
    mask_large_pore = (
        (x - cx_pore) ** 2 / pore_rx**2 + (y - cy_pore) ** 2 / pore_ry**2
    ) <= 1

    base_fluid = (mask_large_pore | mask_channel).astype(np.float64)

    smoothed_fluid = gaussian_filter(base_fluid, sigma=smoothing_radius)

    data[smoothed_fluid > 0.5] = 1

    mask_solid_obstacle = (
        (x - cx_obstacle) ** 2 / obstacle_rx**2
        + (y - cy_obstacle) ** 2 / obstacle_ry**2
    ) <= 1
    data[mask_solid_obstacle] = 0

    semi_width_map = ps.filters.local_thickness(data != 0)
    print(
        f"Double pore maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Double pore minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Double pore mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return data


def create_u_channel(
    length_nodes: int,
    width_nodes: int,
    smoothing_sigma: float = 4.0,
    percentage_radius: float = 0.1,
) -> np.ndarray:
    """Creates a binary image of a smooth U-shaped serpentine channel.

    Args:
        length_nodes: The number of grid nodes along the channel length.
        width_nodes: The number of grid nodes along the channel width.
        smoothing_sigma: Standard deviation for Gaussian kernel to smooth
          the sharp internal corners.
        percentage_radius: The channel radius represented as a fraction of the
          minimum domain dimension.

    Returns:
        A 2D NumPy array of shape (length_nodes, width_nodes) with dtype
        np.float64 where 1 represents the fluid channel and 0 represents the
        solid matrix.
    """
    radius = int(percentage_radius * min(length_nodes, width_nodes))

    skeleton = np.zeros((length_nodes, width_nodes), dtype=bool)
    corner_1 = (int(length_nodes / 4), int(3 * width_nodes / 4))
    corner_2 = (int(length_nodes / 4), int(width_nodes / 4))
    corner_3 = (int(3 * length_nodes / 4), int(width_nodes / 4))
    corner_4 = (int(3 * length_nodes / 4), int(3 * width_nodes / 4))

    skeleton[0 : corner_1[0], corner_1[1]] = True
    skeleton[corner_1[0], corner_2[1] : corner_1[1]] = True
    skeleton[corner_2[0] : corner_3[0], corner_2[1]] = True
    skeleton[corner_3[0], corner_3[1] : corner_4[1]] = True
    skeleton[corner_4[0] :, corner_4[1]] = True

    distance_map = distance_transform_edt(~skeleton)
    fluid_mask = (distance_map < radius).astype(np.float64)

    smoothed_fluid = gaussian_filter(fluid_mask, sigma=smoothing_sigma) > 0.5

    geometry = np.zeros((length_nodes, width_nodes), dtype=np.float64)
    geometry[smoothed_fluid] = 1

    semi_width_map = ps.filters.local_thickness(geometry != 0)
    print(
        f"U channel maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"U channel minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"U channel mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return geometry


def create_curved_channel(domain_size: int) -> np.ndarray:
    """Creates a periodic 45-degree curved channel using circle sectors.

    This function creates quarter-circle channel tracks at the bottom-left
    and top-right corners of a square domain to test grid periodicity.

    Args:
        domain_size: The total number of grid nodes along both X and Y axes
          (creates a square matrix of size domain_size x domain_size).

    Returns:
        A 2D NumPy array of shape (domain_size, domain_size) with dtype
        np.int8, where 1 represents the fluid channel and 0 represents the
        solid boundaries.
    """
    channel_width = 0.2 * domain_size

    inner_radius = domain_size // 2 - channel_width
    outer_radius = domain_size // 2 + channel_width

    geometry = np.zeros((domain_size, domain_size), dtype=np.float64)

    x, y = np.ogrid[:domain_size, :domain_size]

    distance_sq_1 = (x - 0) ** 2 + (y - 0) ** 2
    mask_channel_1 = (distance_sq_1 >= inner_radius**2) & (
        distance_sq_1 < outer_radius**2
    )

    distance_sq_2 = (x - (domain_size - 1)) ** 2 + (y - (domain_size - 1)) ** 2
    mask_channel_2 = (distance_sq_2 >= inner_radius**2) & (
        distance_sq_2 < outer_radius**2
    )

    geometry[mask_channel_1 | mask_channel_2] = 1

    semi_width_map = ps.filters.local_thickness(geometry != 0)
    print(
        f"Curved channel maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Curved channel minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Curved channel mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return geometry


def create_diagonal_square(domain_size: int) -> np.ndarray:
    """Creates a symmetric diamond-like diagonal square channel geometry.

    This function creates a quarter of the geometry, applies a diagonal
    solid boundary cuts, and mirrors it horizontally and vertically to
    create a fully symmetric block.

    Args:
        domain_size: The base size of the quarter-domain matrix.

    Returns:
        A 2D NumPy array of shape (2 * domain_size, 2 * domain_size) with
        dtype np.int8, where 1 represents fluid and 0 represents solid walls.
    """
    square = np.ones((domain_size, domain_size), dtype=np.float64)

    # Vectorized mask to replace the loop step (square[i, i:] = 0)
    i, j = np.ogrid[:domain_size, :domain_size]
    mask = (i < int(0.8 * domain_size)) & (j >= i)
    square[mask] = 0

    # Mirror horizontally and vertically to build the full symmetric domain
    square = np.hstack((np.fliplr(square), square))
    square = np.vstack((square, np.flipud(square)))

    semi_width_map = ps.filters.local_thickness(square != 0)
    print(
        f"Diagonal square maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Diagonal square minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Diagonal square mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return square.T


def create_step_geometry(domain_size: int) -> np.ndarray:
    """Creates a binary image of a channel with a forward/backward step.

    Translates to a classic 'degrau' benchmark configuration where a sudden
    expansion or constriction is formed inside the straight channel.

    Args:
        domain_size: The characteristic size scale used to dimension the step.

    Returns:
        A 2D NumPy array of shape (2 * domain_size, domain_size + 2) with
        dtype np.int8, where 1 represents fluid and 0 represents solid walls.
    """
    # Initialize the domain matrix
    step = np.ones((domain_size * 2, domain_size + 2), dtype=np.float64)

    # Apply external channel walls and the step constriction
    step[:, 0] = 0
    step[:, -1] = 0
    step[:domain_size, : domain_size // 2] = 0

    semi_width_map = ps.filters.local_thickness(step != 0)
    print(
        f"Step maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Step minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Step mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return step


def create_slanted_channel(domain_size: int, width_fraction: float = 0.3) -> np.ndarray:
    """Creas a 2D binary image of a periodic 45-degree slanted channel.

    The domain must be strictly square (NX = NY) to ensure perfect periodicity
    at a 45-degree angle. This configuration is widely used for periodic
    boundary condition benchmarks in fluid dynamics.

    Args:
        domain_size: The number of grid nodes along both X and Y axes (creates a
          square matrix of size domain_size x domain_size).
        width_fraction: The perpendicular channel width represented as a
          fraction of the domain size. Defaults to 0.3.

    Returns:
        A 2D NumPy array of shape (domain_size, domain_size) with dtype
        np.int8, where 1 represents the fluid channel and 0 represents the
        solid boundaries.
    """
    channel_width = width_fraction * domain_size

    i = np.arange(domain_size).reshape(domain_size, 1)
    j = np.arange(domain_size).reshape(1, domain_size)

    offset = domain_size // 2

    diff = (j - i - offset) % domain_size
    diff = np.where(diff > domain_size / 2, diff - domain_size, diff)

    distance = np.abs(diff) / np.sqrt(2.0)

    geometry = np.zeros((domain_size, domain_size), dtype=np.float64)

    geometry[distance < channel_width / 2.0] = 1

    semi_width_map = ps.filters.local_thickness(geometry != 0)
    print(
        f"Slanted channel maximum channel width = {round(np.max(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Slanted channel minimum channel width = {round(np.min(semi_width_map[semi_width_map > 0]) * 2)} voxels"
    )
    print(
        f"Slanted channel mean channel width = {round(np.mean(semi_width_map[semi_width_map > 0]) * 2)} voxels\n"
    )

    return geometry


def calculate_simulation_depths(geometry_mask: np.ndarray) -> np.ndarray:
    """Calculates an array of integer simulation depths based on geometry thickness.

    Extracts the minimum and maximum channel widths from the input geometry using
    a local thickness filter, and generates 7 linearly spaced integer values
    ranging from half the minimum width to double the maximum width.

    Args:
        geometry_mask: A numpy array representing the simulation domain, where
          non-zero values indicate the fluid phase.

    Returns:
        A 1D numpy array containing 7 linearly spaced integer depth values.
    """
    semi_width_map = ps.filters.local_thickness(geometry_mask != 0)

    active_nodes = semi_width_map[semi_width_map > 0]

    min_width = round(np.mean(active_nodes) * 2)
    max_width = round(np.max(active_nodes) * 2)

    start_depth = min(min_width / 2.0, 8)
    end_depth = max_width * 4.0

    depth_values = np.round(np.linspace(start_depth, end_depth, 9)).astype(np.float64)

    return depth_values
