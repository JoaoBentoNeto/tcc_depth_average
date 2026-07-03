import numpy as np


def calculate_permeability_3d_rectangular_duct(width: float, height: float) -> float:
    """Calculates the analytical 3D permeability of a rectangular duct.

    Args:
        width: The width of the channel.
        height: The height of the channel.

    Returns:
        The calculated 3D permeability value.
    """
    series_sum = 0.0

    for n in range(1, 2000, 2):
        term = (1.0 / n**4) * (
            1.0
            - (height / (n * np.pi * width / 2.0))
            * np.tanh(n * np.pi * width / 2.0 / height)
        )
        series_sum += term

    k_3d = (8.0 * height**2 / np.pi**4) * series_sum

    return k_3d


def calculate_permeability_2p5d_rectangular_duct(width: float, height: float) -> float:
    """Calculates the depth-averaged (2.5D) permeability approximation.

    Args:
        width: The width of the channel.
        height: The height of the channel.

    Returns:
        The calculated 2.5D permeability value.
    """
    r = np.sqrt(12.0) / height

    k_2p5d = (height**2 / 12.0) * (1.0 - np.tanh(r * width / 2.0) / (r * width / 2.0))

    return k_2p5d


def calculate_analytical_velocity_3d_rectangular_duct(
    x_coords: np.ndarray | float,
    z_coords: np.ndarray | float,
    height: float,
    width: float,
    body_force: float,
    tau: float,
    length_nodes: int,
    flow_axis: int = 1,
) -> np.ndarray | float:
    """Calculate the 3D analytical velocity profile for a rectangular duct.

    Args:
        x_coords: Coordinate(s) along the width axis. Can be a float or array.
        z_coords: Coordinate(s) along the depth axis. Can be a float or array.
        height: The total height of the channel.
        width: The total width of the channel.
        body_force: The driving acceleration (gravity or pressure gradient).
        tau: The LBM relaxation time, used to calculate kinematic viscosity.
        length_nodes: The number of grid nodes along the channel length to stretch
            the 2D profile into 3D. If None, returns the 2D cross-section.
        flow_axis: The axis index representing the flow direction (default is 1).

    Returns:
        The analytical velocity evaluated at the given coordinates, stretched
        to 3D if length_nodes is provided.
    """
    cs2 = 1.0 / 3.0
    kinematic_viscosity = (tau - 0.5) * cs2
    base_factor = (4.0 * height**2 * body_force) / (np.pi**3 * kinematic_viscosity)

    n = np.arange(1, 5000, 2)

    x_ext = np.atleast_1d(x_coords)[..., None]
    z_ext = np.atleast_1d(z_coords)[..., None]

    a = n * np.pi * np.abs(x_ext) / height
    b = n * np.pi * width / (2.0 * height)

    cosh_ratio = np.exp(a - b) * (1.0 + np.exp(-2.0 * a)) / (1.0 + np.exp(-2.0 * b))

    term1 = 1.0 - cosh_ratio
    term2 = np.sin(n * np.pi * z_ext / height)
    term3 = n**3

    summation = np.sum(term1 * term2 / term3, axis=-1)

    result = base_factor * summation

    if np.isscalar(x_coords) and np.isscalar(z_coords):
        return float(result.item())

    result = np.expand_dims(result, axis=flow_axis)

    target_shape = list(result.shape)
    target_shape[flow_axis] = length_nodes

    result = np.broadcast_to(result, target_shape)

    return result


def calculate_analytical_velocity_map_3d_rectangular_duct(
    x_coords: np.ndarray | float,
    height: float,
    width: float,
    body_force: float,
    tau: float,
    length_nodes: int,
    flow_axis: int = 1,
) -> np.ndarray | float:
    """Calculate the depth averaged 3D analytical velocity profile for a rectangular duct.

    Args:
        x_coords: Coordinate(s) along the width axis. Can be a float or array.
        height: The total height of the channel.
        width: The total width of the channel.
        body_force: The driving acceleration (gravity or pressure gradient).
        tau: The LBM relaxation time, used to calculate kinematic viscosity.
        length_nodes: The number of grid nodes along the channel length to stretch
            the 2D profile into 3D. If None, returns the 2D cross-section.
        flow_axis: The axis index representing the flow direction (default is 1).

    Returns:
        The analytical velocity evaluated at the given coordinates, stretched
        to 3D if length_nodes is provided.
    """
    cs2 = 1.0 / 3.0
    kinematic_viscosity = (tau - 0.5) * cs2
    base_factor = (8.0 * height**2 * body_force) / (np.pi**4 * kinematic_viscosity)

    n = np.arange(1, 5000, 2)

    x_ext = np.atleast_1d(x_coords)[..., None]

    a = n * np.pi * np.abs(x_ext) / height
    b = n * np.pi * width / (2.0 * height)

    cosh_ratio = np.exp(a - b) * (1.0 + np.exp(-2.0 * a)) / (1.0 + np.exp(-2.0 * b))

    term1 = 1.0 - cosh_ratio
    term3 = n**4

    summation = np.sum(term1 / term3, axis=-1)

    result = base_factor * summation
    # print(f"result shape 1 {result.shape}")

    if np.isscalar(x_coords):
        return float(result.item())

    result = np.pad(result, pad_width=(1, 1), mode="constant", constant_values=0.0)
    # print(f"result shape 2 {result.shape}")

    result = np.expand_dims(result, axis=flow_axis)
    # print(f"result shape 3 {result.shape}")
    target_shape = list(result.shape)
    target_shape[flow_axis] = length_nodes

    result = np.broadcast_to(result, target_shape)

    return result


def calculate_analytical_velocity_map_2p5d_rectangular_duct(
    x_coords: np.ndarray | float,
    height: float,
    width: float,
    body_force: float,
    tau: float,
    length_nodes: int,
    flow_axis: int = 1,
) -> np.ndarray | float:
    """Calculate the 2.5D analytical velocity profile for a rectangular duct.

    Args:
        x_coords: Coordinate(s) along the width axis. Can be a float or array.
        height: The total height of the channel.
        width: The total width of the channel.
        body_force: The driving acceleration (gravity or pressure gradient).
        tau: The LBM relaxation time, used to calculate kinematic viscosity.
        length_nodes: The number of grid nodes along the channel length to stretch
            the 2D profile into 3D. If None, returns the 2D cross-section.
        flow_axis: The axis index representing the flow direction (default is 1).

    Returns:
        The analytical velocity evaluated at the given coordinates, stretched
        to 3D if length_nodes is provided.
    """
    cs2 = 1.0 / 3.0
    kinematic_viscosity = (tau - 0.5) * cs2
    base_factor = height**2 * body_force / (12.0 * kinematic_viscosity)
    # print(f"x shape 0 {x_coords.shape}")
    # x_ext = np.atleast_1d(x_coords)[..., None]
    x_ext = x_coords
    # print(f"x shape 1 {x_ext.shape}")

    a = np.cosh(np.sqrt(12.0) * x_ext / height)
    b = np.cosh(np.sqrt(12.0) * width / height / 2)

    result = base_factor * (1.0 - a / b)

    # print(f"result shape 1 {result.shape}")

    if np.isscalar(x_coords):
        return float(result.item())

    result = np.pad(result, pad_width=(1, 1), mode="constant", constant_values=0.0)
    # print(f"result shape 2 {result.shape}")
    result = np.expand_dims(result, axis=flow_axis)
    # print(f"result shape 3 {result.shape}")
    target_shape = list(result.shape)
    target_shape[flow_axis] = length_nodes

    result = np.broadcast_to(result, target_shape)

    return result


def calculate_permeability_parallel_plates(height: float) -> float:
    """Calculates the analytical permeability of a Hele-Shaw cell.

    Args:
        height: The height of the channel.

    Returns:
        The calculated permeability value.
    """
    return height**2 / 12.0


def calculate_analytical_velocity_map_parallel_plates(
    length_nodes: int,
    width_nodes: int,
    height_nodes: int,
    body_force: float,
    tau: float,
) -> np.ndarray:
    """Calculate the analytical mean velocity map for a Hele-Shaw cell (parallel plates).

    Args:
        length_nodes: The number of grid nodes along the channel length.
        width_nodes: The number of grid nodes along the channel width.
        height_nodes: The number of grid nodes along the channel height (gap) between the parallel plates.
        body_force: The driving acceleration (gravity or kinematic pressure gradient).
        tau: The LBM relaxation time, used to calculate kinematic viscosity.

    Returns:
        A 2D array with the constant mean velocity evaluated for the whole domain.
    """
    cs2 = 1.0 / 3.0
    kinematic_viscosity = (tau - 0.5) * cs2

    mean_velocity = (body_force * height_nodes**2) / (12.0 * kinematic_viscosity)

    result = np.ones((length_nodes, width_nodes), dtype=np.float64) * mean_velocity

    return result


if __name__ == "__main__":
    height = 100
    aspect_ratio = 10
    resolution = 0.5
    width = round(height / aspect_ratio / resolution)

    print(f"width = {width} voxel = {width * resolution} um")

    k = calculate_permeability_3d_rectangular_duct(
        width=width * resolution, height=height
    )
    print(f"absperm = {k * width / (width + 2)} um^2")
