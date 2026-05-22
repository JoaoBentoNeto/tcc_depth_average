import os
import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import pyvista as pv
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize as ski_skeletonize


def read_db(
    db_path: str,
) -> tuple[list[int], float, float, str | None, str | None, float, int, list[float]]:
    """Reads and parses the simulation database configuration file.

    Args:
        db_path: The absolute or relative path to the .db file.

    Returns:
        A tuple containing the extracted simulation parameters:
        - domain: A list of integers representing the dimensions.
        - resolution: The conversion factor in micrometers/voxel.
        - tau: The relaxation time.
        - simulation_path: The directory path (None if not found).
        - simulation_name: The name of the simulation job (None if not found).
        - tolerance: The convergence tolerance parameter.
        - timestepmax: The maximum number of timesteps.
        - body_force: A list of floats representing the acceleration field.
    """
    parsed_data = {}

    with open(db_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')
            parsed_data[key] = value

    tau = float(parsed_data["tau"])
    body_force = [float(v.strip()) for v in parsed_data["F"].split(",")]
    timestepmax = int(parsed_data["timestepMax"])
    tolerance = float(parsed_data["tolerance"])

    simulation_path = parsed_data.get("simulation_path", None)
    simulation_name = parsed_data.get("simulation_name", None)
    simulation_type = parsed_data.get("simulation_type", None)

    resolution = float(parsed_data["voxel_length"])
    domain = [int(v.strip()) for v in parsed_data["N"].split(",")]

    return (
        domain,
        resolution,
        tau,
        simulation_path,
        simulation_name,
        tolerance,
        timestepmax,
        body_force,
        simulation_type,
    )


def export_vti(
    ux: np.ndarray,
    uy: np.ndarray,
    rho: np.ndarray,
    resolution: float,
    final_timestep: int,
    simulation_path: str,
) -> None:
    """Exports the simulated fields to VTI format, cloning the LBPM structure.

    Args:
        ux: A 2D NumPy array representing the velocity in the X direction.
        uy: A 2D NumPy array representing the velocity in the Y direction.
        rho: A 2D NumPy array representing the fluid density.
        resolution: The physical size of each voxel.
        final_timestep: The last time step, used to create the output folder.
        simulation_path: The base directory path where the output is saved.

    Returns:
        None. The grid is saved directly to the file system.
    """
    vis_name = f"vis{final_timestep}"
    vis_path = os.path.join(simulation_path, vis_name)
    os.makedirs(vis_path, exist_ok=True)

    output_path = os.path.join(vis_path, "summary.vti")

    nx_cells, ny_cells = ux.shape
    nz_cells = 1

    grid = pv.ImageData()

    # Dimensions mapped as (Y, Z, X) to match the LBPM coordinate system
    grid.dimensions = (ny_cells + 1, nz_cells + 1, nx_cells + 1)
    grid.spacing = (resolution, resolution, resolution)

    grid.cell_data["Velocity_x"] = uy.flatten(order="F")
    grid.cell_data["Velocity_y"] = np.zeros_like(ux).flatten(order="F")
    grid.cell_data["Velocity_z"] = ux.flatten(order="F")
    grid.cell_data["Density"] = rho.flatten(order="F")

    grid.save(output_path)


def calculate_semi_width_map(depth_map: np.ndarray) -> np.ndarray:
    """Calculates the semi-width map of a fluid domain.

    Applies a combination of distance transform and skeletonization with
    a maximum filter to accurately map the semi-width of complex fluid channels.
    Uses dynamic periodic padding determined by the maximum radius to guarantee
    perfect skeletons spanning the boundaries and aligning with the flow axis.

    Args:
        depth_map: A numpy array (2D or batched N-D) representing the domain,
            where 0 indicates solid and non-zero values indicate fluid.

    Returns:
        A numpy array of the same shape as `depth_map` containing the calculated
        semi-width values for the fluid pixels, and 0 for the solid pixels.
    """
    fluid_initial = depth_map != 0

    # O raio máximo dita a espessura da geometria
    max_radius = int(np.max(distance_transform_edt(fluid_initial)))

    # PAD LIVRE DE TRAVAS: Exatamente o raio + margem de segurança.
    # Remover o limite de dim_size permite que eixos finos (como Y=4)
    # cresçam o suficiente para que o esqueleto se alinhe corretamente.
    pad_size = max_radius * 2 + 5

    pad_axes = (0, 1)
    pad_width = [(pad_size, pad_size) if i in pad_axes else (0, 0) for i in range(depth_map.ndim)]

    # mode="wrap" garante a continuidade periódica da simulação
    padded_depth_map = np.pad(depth_map, pad_width, mode="wrap")

    fluid = padded_depth_map != 0
    del padded_depth_map

    distance_map = distance_transform_edt(fluid)
    skeleton = ski_skeletonize(fluid)

    # plt.imshow(skeleton)
    # plt.axis('on')
    # plt.colorbar()
    # plt.show()

    max_dist = maximum_filter(distance_map, size=3)
    distance_map[skeleton] = max_dist[skeleton]
    del max_dist

    indices = distance_transform_edt(skeleton == 0, return_indices=True)[1]
    del skeleton

    distance_map = distance_map[tuple(indices)]
    del indices

    distance_map *= fluid
    del fluid

    slices = [slice(pad[0], -pad[1]) if pad[0] > 0 else slice(None) for pad in pad_width]

    return distance_map[tuple(slices)].copy()


def calculate_shear_permeability(
    semi_width_map: np.ndarray,
    depth_map: np.ndarray | float,
) -> np.ndarray:
    """Calculates the optimized shear permeability map for a fluid domain.

    Uses a memory-efficient loop over odd wave numbers to compute
    the local permeability based on the channel's semi-width and depth,
    computing only the active fluid nodes.

    Args:
        semi_width_map: A 2D numpy array representing the local semi-width
            of the channel.
        depth_map: A 2D numpy array or float representing the local depth
            of the channel.

    Returns:
        A numpy array containing the computed permeability values for the fluid
        regions, and 1.0 for the solid regions.
    """
    fluid_mask = semi_width_map != 0

    distance_map = distance_transform_edt(fluid_mask)

    dist = distance_map[fluid_mask]
    width = semi_width_map[fluid_mask]

    if isinstance(depth_map, np.ndarray):
        d = depth_map[fluid_mask]
    else:
        d = depth_map

    del distance_map

    num_sum = np.zeros_like(dist, dtype=np.float64)
    den_sum = np.zeros_like(dist, dtype=np.float64)

    for n_val in range(1, 2500, 2):
        n = float(n_val)

        pi_n_over_d = (np.pi * n) / d

        arg1 = -pi_n_over_d * dist
        arg2 = -pi_n_over_d * (2.0 * width - dist)
        arg3 = -pi_n_over_d * 2.0 * width

        ratio = (np.exp(arg1) + np.exp(arg2)) / (1.0 + np.exp(arg3))

        n2 = n * n
        n4 = n2 * n2
        den_sum += ratio / n2
        num_sum += ratio / n4

    numerator = (np.pi**4 / 96.0) - num_sum
    denominator = (np.pi**2 / 8.0) - den_sum

    perm_fluid = (d**2 / np.pi**2) * (numerator / denominator)

    permeability = np.ones_like(semi_width_map, dtype=np.float64)

    permeability[fluid_mask] = perm_fluid

    return permeability


def run_lbm(db_path: str) -> None:
    """Executes a depth-averaged LBM simulation based on Laleian et al. (2015).

    Args:
        db_path: The absolute or relative path to the .db configuration file.

    Returns:
        None. Displays terminal logs and generates .vti files.
    """
    (
        domain,
        resolution,
        tau,
        simulation_path,
        simulation_name,
        tolerance,
        timestepmax,
        body_force,
        simulation_type,
    ) = read_db(db_path=db_path)

    print(
        f"\nSimulation Name = {simulation_name}\n"
        f"Domain = {domain}\n"
        f"Resolution = {resolution}\n"
        f"Tau = {tau}\n"
        f"Simulation Path = {simulation_path}\n"
        f"Tolerance = {tolerance}\n"
        f"Timestep Max = {timestepmax}\n"
        f"Body Force = {body_force}\n"
    )

    nx, ny = domain
    raw_path = os.path.join(simulation_path, f"{simulation_name}.raw")
    depth_map = np.fromfile(raw_path, dtype=float).reshape(domain)
    solid_mask = depth_map == 0.0
    safe_depth = np.where(solid_mask[:, :, None], 1.0, depth_map[:, :, None] / resolution)
    max_depth = np.max(depth_map / resolution)
    print(f"Maximum Depth (h_max)= {max_depth} voxels= {max_depth * resolution} \u03bcm\n")
    cs2 = 1.0 / 3.0
    nu = (tau - 0.5) * cs2
    if simulation_type == "laleian2015":
        drag_resistance = nu / (safe_depth**2 / 12.0)
    elif simulation_type == "rdm":
        # semi_width_map = calculate_semi_width_map(depth_map=depth_map)
        semi_width_map = ps.filters.local_thickness(~solid_mask)
        print(
            f"Maximum channel width = {np.max(semi_width_map) * 2} voxels = {np.max(semi_width_map) * 2 * resolution}  \u03bcm"
        )
        print(
            f"Minimum channel width = {np.min(semi_width_map) * 2} voxels = {np.min(semi_width_map) * 2 * resolution}  \u03bcm"
        )
        print(
            f"Mean channel width = {np.mean(semi_width_map) * 2} voxels = {np.mean(semi_width_map) * 2 * resolution}  \u03bcm\n"
        )
        plt.imshow(semi_width_map)
        plt.axis("on")
        plt.colorbar()
        plt.savefig(
            os.path.join(simulation_path, f"{simulation_name}.png"), dpi=1000, bbox_inches="tight", pad_inches=0.1
        )
        plt.close()
        perm = calculate_shear_permeability(semi_width_map=semi_width_map, depth_map=depth_map)
        drag_resistance = nu / (perm[:, :, None] / resolution**2)

    # convenção das velocidades
    #
    #         8   3   5       y
    #          \  |  /        |
    #           \ | /         |
    #      2 ---  0  --- 1    L --- x
    #           / | \
    #          /  |  \
    #         6   4   7

    e_x = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1])
    e_y = np.array([0, 0, 0, 1, -1, 1, -1, -1, 1])
    e_x_3d = e_x[None, None, :]
    e_y_3d = e_y[None, None, :]
    opposite_dir = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
    weights = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
    weights_3d = weights[None, None, :]
    rho = np.ones((nx, ny, 1))
    f = weights_3d * rho
    f_post = np.zeros_like(f)
    uh_x = np.zeros((nx, ny, 1))
    uh_y = np.zeros((nx, ny, 1))
    force_x = np.zeros((nx, ny, 1))
    force_y = np.zeros((nx, ny, 1))

    bounce_back_mask = np.zeros((nx, ny, 9), dtype=bool)
    for k in range(9):
        solid_neighbor = np.roll(solid_mask, shift=(-e_x[k], -e_y[k]), axis=(0, 1))
        bounce_back_mask[:, :, k] = (~solid_mask) & solid_neighbor

    error = 1e10
    previous_velocity_sum = 1e10
    timestep = 0

    while error > tolerance and timestep < timestepmax:
        # --- Macroscopic Variables ---
        rho = np.sum(f, axis=-1, keepdims=True)
        mom_x = np.dot(f, e_x)[:, :, None]
        mom_y = np.dot(f, e_y)[:, :, None]
        uh_x = (mom_x + 0.5 * safe_depth * body_force[0]) / (1.0 + 0.5 * drag_resistance)
        uh_y = (mom_y + 0.5 * safe_depth * body_force[1]) / (1.0 + 0.5 * drag_resistance)
        force_x = safe_depth * body_force[0] - (drag_resistance * uh_x)
        force_y = safe_depth * body_force[1] - (drag_resistance * uh_y)
        u_squared = uh_x**2 + uh_y**2

        # --- Collision and Forcing ---
        e_dot_u = e_x_3d * uh_x + e_y_3d * uh_y
        f_eq = weights_3d * rho * (1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u**2 - 1.5 * u_squared)
        term_x = (e_x_3d - uh_x) * 3.0 + e_dot_u * 9.0 * e_x_3d
        term_y = (e_y_3d - uh_y) * 3.0 + e_dot_u * 9.0 * e_y_3d
        forcing = weights_3d * (1.0 - 0.5 / tau) * (term_x * force_x + term_y * force_y)
        f_post = f - (1.0 / tau) * (f - f_eq) + forcing

        # --- Streaming / Propagation ---
        for k in range(9):
            f[:, :, k] = np.roll(f_post[:, :, k], shift=(e_x[k], e_y[k]), axis=(0, 1))

        # --- Bounce-Back Boundary Condition ---
        for k in range(9):
            mask = bounce_back_mask[:, :, k]
            f[mask, opposite_dir[k]] = f_post[mask, k]

        # --- Convergence Check ---
        timestep += 1
        if timestep % 100 == 0:
            current_velocity_sum = np.sum(np.sqrt(u_squared))
            error = np.abs(current_velocity_sum - previous_velocity_sum) / current_velocity_sum

            previous_velocity_sum = current_velocity_sum
            print(f"Timestep {timestep}: error = {error:.1e}", end="\r")

    u_x = np.where(solid_mask, 0.0, uh_x[:, :, 0] / max_depth)
    u_y = np.where(solid_mask, 0.0, uh_y[:, :, 0] / max_depth)
    rho_2d = rho[:, :, 0]
    um_to_md = 0.0009869233
    abs_perm = resolution**2 * np.mean(u_x) * nu / body_force[0]
    print(f"\n\nAbsolute Permeability = {abs_perm:.4f} \u03bcm^2 = {abs_perm / um_to_md:.4f} mD")

    # expected_3d = (
    #     analytical.calculate_permeability_3d_rectangular_duct(width=domain[1] - 2, height=max_depth)
    #     * resolution**2
    #     * (domain[1] - 2)
    #     / domain[1]
    # )
    # print(f"\n\nExpected 3D Permeability = {expected_3d:.4f} \u03bcm^2 = {expected_3d / um_to_md:.4f} mD")
    # print(f"error 3D = {abs(abs_perm - expected_3d) / expected_3d * 100} %")
    # expected_2p5d = (
    #     analytical.calculate_permeability_2p5d_rectangular_duct(width=domain[1] - 2, height=max_depth)
    #     * resolution**2
    #     * (domain[1] - 2)
    #     / domain[1]
    # )
    # print(f"\n\nExpected 2.5D Permeability = {expected_2p5d:.4f} \u03bcm^2 = {expected_2p5d / um_to_md:.4f} mD")
    # print(f"error 2.5D = {abs(abs_perm - expected_2p5d) / expected_2p5d * 100} %")

    # expected = (
    #     analytical.calculate_permeability_parallel_plates(width=domain[1] - 2, height=max_depth) * resolution**2
    # )
    # print(f"\n\nExpected Permeability = {expected:.4f} \u03bcm^2 = {expected / um_to_md:.4f} mD")
    # print(f"error = {(abs_perm - expected) / expected * 100} %")

    export_vti(
        ux=u_x,
        uy=u_y,
        rho=rho_2d,
        resolution=resolution,
        final_timestep=timestep,
        simulation_path=simulation_path,
    )

    print("\nSimulation Finished Successfully!")


if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 2:
        print("ERROR: The .db file path was not provided.")
        print("Usage: python laleian2015_run.py /path/to/file.db")
        sys.exit(1)

    db_path = sys.argv[1]

    print(f"Starting simulation with database: {db_path}")

    run_lbm(db_path)

    end_time = time.time()
    duration_seconds = end_time - start_time
    duration = timedelta(seconds=duration_seconds)
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTotal execution time:{days}d-{hours:02d}:{minutes:02d}:{seconds:02d}")
