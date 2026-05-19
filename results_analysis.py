import glob
import os
import re
import traceback
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

import run_laleian2015


def read_vti(
    simulation_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Reads 3D velocity fields from the latest visualization folder.

    Args:
        simulation_path: The base directory of the simulation containing
            the 'vis*' output folders.

    Returns:
        A tuple containing four elements:
        - vx_3d: A 3D NumPy array of velocities in the X direction.
        - vy_3d: A 3D NumPy array of velocities in the Y direction.
        - vz_3d: A 3D NumPy array of velocities in the Z direction.
        - shape: A tuple representing the dimensions of the cell data

    Raises:
        FileNotFoundError: If no 'vis*' directories are found in the path.
    """
    vis_pattern = os.path.join(simulation_path, "vis*")
    vis_folders = glob.glob(vis_pattern)

    if not vis_folders:
        raise FileNotFoundError(
            f"No 'vis*' directory found in {simulation_path}."
        )

    # Sorts the list and selects the last folder (latest timestep)
    vis_folders.sort()
    latest_vis_folder = vis_folders[-1]

    grid_file = os.path.join(latest_vis_folder, "summary.pvti")

    if not os.path.exists(grid_file):
        grid_file = os.path.join(latest_vis_folder, "summary.vti")

    grid = pv.read(grid_file)
    nx, ny, nz = grid.dimensions
    shape = (nx - 1, ny - 1, nz - 1)

    vx_3d = grid.cell_data["Velocity_x"].reshape(shape, order="F")
    vy_3d = grid.cell_data["Velocity_y"].reshape(shape, order="F")
    vz_3d = grid.cell_data["Velocity_z"].reshape(shape, order="F")

    return vx_3d, vy_3d, vz_3d, shape


def get_depth_averaged_map(
    simulation_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the depth-averaged 2D velocity fields from 3D data.

    Reads the 3D velocity fields and averages the horizontal components and
    magnitude across the depth axis. Excludes the solid boundary wall layers
    (top and bottom) from the average if the depth is greater than 1 voxel.

    Args:
        simulation_path: The base directory of the simulation containing
            the visualization outputs.

    Returns:
        A tuple containing three 2D NumPy arrays:
        - vz_map: The depth-averaged velocity in the Z direction.
        - vx_map: The depth-averaged velocity in the X direction.
        - mag_map: The depth-averaged 3D velocity magnitude.
    """
    vx_3d, vy_3d, vz_3d, shape = read_vti(simulation_path=simulation_path)

    mag_3d = np.sqrt(vx_3d**2 + vy_3d**2 + vz_3d**2)

    if shape[1] > 1:
        vx_3d = vx_3d[:, 1:-1, :]
        vz_3d = vz_3d[:, 1:-1, :]
        mag_3d = mag_3d[:, 1:-1, :]

    vz_map = np.mean(vz_3d, axis=1)
    vx_map = np.mean(vx_3d, axis=1)
    mag_map = np.mean(mag_3d, axis=1)

    return vz_map, vx_map, mag_map


def calculate_permeability(
    streamwise_velocity: np.ndarray,
    tau: float,
    body_force: float,
    resolution: float,
) -> float:
    """Calculates the absolute permeability from LBM simulation results.

    Uses Darcy's law adapted for lattice units to compute the permeability
    based on the mean velocity in the flow direction.

    Args:
        streamwise_velocity: Velocity field array in the main flow direction.
        tau: The relaxation time parameter used in the simulation.
        body_force: The applied external body force driving the fluid.
        resolution: The physical resolution (size) of a single voxel.

    Returns:
        The calculated absolute permeability as a float.
    """
    kinematic_viscosity = (tau - 0.5) / 3.0

    abs_perm = (
        (resolution**2)
        * np.mean(streamwise_velocity)
        * kinematic_viscosity
        / body_force
    )

    return float(abs_perm)


def calculate_velocity_errors(
    test_velocity: np.ndarray | None,
    ref_velocity: np.ndarray | None,
    test_coords: tuple[np.ndarray, np.ndarray] | None = None,
    ref_coords: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float, np.ndarray]:
    """Computes velocity errors between test and reference data.

    If coordinate grids are provided and differ, the reference velocity is
    interpolated onto the test coordinates. Calculates the normalized RMSE
    according to Equation 15 from Laleian et al. (2015).

    Args:
        test_velocity: 2D NumPy array of the test velocity field, or None
            if is a experimental result without velocity field.
        ref_velocity: 2D NumPy array of the reference velocity field, or None
            if is a experimental result without velocity field.
        test_coords: A tuple of 2D arrays (X, Y) for the test physical coordinates.
        ref_coords: A tuple of 2D arrays (X, Y) for the reference physical coordinates.

    Returns:
        A tuple containing:
        - normalized_rmse: The normalized root-mean-square error of the velocity.
        - mean_velocity_error: The mean local absolute percentage error.
        - velocity_error_map: 2D array of the local absolute percentage error.
    """
    if ref_velocity is None:
        return float("nan"), float("nan"), np.full_like(test_velocity, np.nan)
    elif test_velocity is None:
        return float("nan"), float("nan"), np.full_like(ref_velocity, np.nan)

    if test_coords is not None and ref_coords is not None:
        shapes_differ = test_velocity.shape != ref_velocity.shape
        coords_differ = not (
            np.array_equal(test_coords[0], ref_coords[0])
            and np.array_equal(test_coords[1], ref_coords[1])
        )

        if shapes_differ or coords_differ:
            points = np.column_stack(
                (ref_coords[0].ravel(), ref_coords[1].ravel())
            )
            values = ref_velocity.ravel()

            ref_velocity = griddata(
                points=points,
                values=values,
                xi=(test_coords[0], test_coords[1]),
                method="linear",
                fill_value=0.0,
            )

    fluid_mask = np.abs(ref_velocity) > 1e-14
    velocity_error_map = np.zeros_like(test_velocity)
    n_liquid_nodes = np.count_nonzero(fluid_mask)

    if n_liquid_nodes > 0:
        velocity_error_map[fluid_mask] = (
            np.abs(ref_velocity[fluid_mask] - test_velocity[fluid_mask])
            / np.abs(ref_velocity[fluid_mask])
            * 100.0
        )

        mean_velocity_error = float(np.mean(velocity_error_map[fluid_mask]))

        rmse = np.sqrt(
            np.sum((test_velocity - ref_velocity) ** 2) / n_liquid_nodes
        )

        abs_ref = np.abs(ref_velocity[fluid_mask])
        velocity_range = np.max(abs_ref) - np.min(abs_ref)

        normalized_rmse = rmse / velocity_range if velocity_range > 0 else rmse
    else:
        mean_velocity_error = float("nan")
        normalized_rmse = float("nan")
    return normalized_rmse, mean_velocity_error, velocity_error_map


def calculate_permeability_errors(
    test_permeability: float,
    ref_permeability: float,
) -> float:
    """Computes the absolute percentage error in permeability.

    Args:
        test_permeability: The computed permeability from the test model.
        ref_permeability: The reference permeability from analytical,
            experimental, or 3D model sources.

    Returns:
        The absolute percentage error in permeability.
    """
    permeability_error = (
        abs(test_permeability - ref_permeability) / ref_permeability * 100.0
    )

    return permeability_error


def extract_simulation_data(parent_folder_path: str) -> dict:
    """ExtractsLBM simulation results from a directory tree.

    Iterates through simulation directories, reads database files, computes
    depth-averaged maps, and calculates the absolute permeability for each run.

    Args:
        parent_folder_path: The root directory containing the simulation cases.

    Returns:
        A nested dictionary containing the results organized by
        simulation type and identifier.
    """
    valid_cases = ["fullscale", "greyscale", "2D", "laleian2015"]
    simulation_data = {case: {} for case in valid_cases}

    for case in valid_cases:
        case_path = os.path.join(parent_folder_path, case)
        if not os.path.isdir(case_path):
            continue

        sim_folders = [
            f
            for f in os.listdir(case_path)
            if os.path.isdir(os.path.join(case_path, f))
        ]

        for sim_folder in sim_folders:
            sim_path = os.path.join(case_path, sim_folder)

            match = re.search(r"_([a-zA-Z]+_\d+(?:p\d+)?)$", sim_folder)
            identifier = match.group(1) if match else sim_folder

            db_path = os.path.join(sim_path, f"{sim_folder}.db")
            if not os.path.exists(db_path):
                print(
                    f"Warning: .db file not found for {sim_folder}. Skipping..."
                )
                continue

            try:
                (_, resolution, tau, _, _, _, _, body_force) = (
                    run_laleian2015.read_db(db_path)
                )

                vz_map, _, _ = get_depth_averaged_map(sim_path)
            except Exception as e:
                print(f"Warning: Could not process data for {sim_folder}: {e}")
                traceback.print_exc()
                continue

            driving_force = max(body_force)
            flow_velocity = vz_map

            sim_permeability = calculate_permeability(
                streamwise_velocity=flow_velocity,
                tau=tau,
                body_force=driving_force,
                resolution=resolution,
            )

            nx, ny = flow_velocity.shape
            x_test, y_test = np.meshgrid(
                np.arange(nx) * resolution,
                np.arange(ny) * resolution,
                indexing="ij",
            )

            simulation_data[case][identifier] = {
                "permeability": sim_permeability,
                "velocity": flow_velocity,
                "coords": (x_test, y_test),
            }

    return simulation_data


def compute_errors(
    simulation_data: dict,
    analytical_data: dict[str, dict[str, dict]] | None = None,
    experimental_data: dict[str, float] | None = None,
) -> dict:
    """Computes simulation errors.

    Args:
        simulation_data: A nested dictionary of extracted simulation data.
        analytical_data: A nested dictionary mapping ["3d" or "2p5d"] ->
            [identifier] -> dict containing permeability, velocity, and coords.
        experimental_data: A dict mapping the identifier to k_experimental.

    Returns:
        A nested dictionary containing the computed errors for each simulation.
    """
    results = {case: {} for case in simulation_data.keys()}

    for case, runs in simulation_data.items():
        for identifier, data in runs.items():
            test_perm = data["permeability"]
            test_vel = data["velocity"]
            test_coords = data["coords"]

            ref_perm = None
            if (
                experimental_data is not None
                and identifier in experimental_data
            ):
                ref_perm = experimental_data[identifier]
            elif (
                analytical_data is not None
                and "3d" in analytical_data
                and identifier in analytical_data["3d"]
            ):
                ref_perm = analytical_data["3d"][identifier]["permeability"]
            elif case != "fullscale" and identifier in simulation_data.get(
                "fullscale", {}
            ):
                ref_perm = simulation_data["fullscale"][identifier][
                    "permeability"
                ]

            ref_vel = None
            ref_coords = None
            if (
                analytical_data is not None
                and "3d" in analytical_data
                and identifier in analytical_data["3d"]
            ):
                ref_vel = analytical_data["3d"][identifier]["velocity"]
                ref_coords = analytical_data["3d"][identifier]["coords"]
            elif case != "fullscale" and identifier in simulation_data.get(
                "fullscale", {}
            ):
                ref_vel = simulation_data["fullscale"][identifier]["velocity"]
                ref_coords = simulation_data["fullscale"][identifier]["coords"]

            if ref_perm is not None:
                perm_error = calculate_permeability_errors(
                    test_permeability=test_perm,
                    ref_permeability=ref_perm,
                )
            else:
                perm_error = float("nan")

            if ref_vel is not None:
                rmse, mean_vel_error, _ = calculate_velocity_errors(
                    test_velocity=test_vel,
                    ref_velocity=ref_vel,
                    test_coords=test_coords,
                    ref_coords=ref_coords,
                )
            else:
                rmse = float("nan")
                mean_vel_error = float("nan")

            results[case][identifier] = {
                "permeability": test_perm,
                "permeability_error": perm_error,
                "normalized_rmse": rmse,
                "mean_velocity_error": mean_vel_error,
            }

    return results


def plot_permeability_errors(
    parent_folder_path: str,
    simulation_name: str,
    results: dict,
    x_label: str,
    x_scale: Literal["linear", "log"] = "linear",
    expected_error: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Plots and saves the permeability errors from simulation results.

    Args:
        parent_folder_path: The root directory containing the simulation cases.
        simulation_name: The base name used to save the output image file.
        results: A nested dictionary containing compiled simulation errors.
        x_label: The label for the X-axis (e.g., "Aspect Ratio (AR)").
        x_scale: The scale for the X-axis, either "linear" or "log".
        expected_error: An optional tuple of (X_values, Y_values) representing
            the expected analytical error to be plotted as a continuous line.

    Returns:
        None. Displays the matplotlib plot and saves it to disk.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    styles = {
        "fullscale": {"color": "#1f77b4", "marker": "o"},
        "greyscale": {"color": "#ff7f0e", "marker": "s"},
        "2D": {"color": "#2ca02c", "marker": "^"},
        "laleian2015": {"color": "#d62728", "marker": "D"},
    }

    legend_labels = {
        "fullscale": "Fullscale 3D",
        "greyscale": "Greyscale (2.5D)",
        "2D": "Standard 2D",
        "laleian2015": "Laleian et al. (2015)",
    }

    for case, runs in results.items():
        x_vals = []
        y_vals = []

        for identifier, data in runs.items():
            perm_error = data.get("permeability_error", float("nan"))

            if np.isnan(perm_error):
                continue

            match = re.search(r"(\d+(?:p\d+)?)", identifier)
            if match:
                num_str = match.group(1).replace("p", ".")
                x_vals.append(float(num_str))
                y_vals.append(perm_error)

        if not x_vals:
            continue

        sorted_pairs = sorted(zip(x_vals, y_vals))
        x_vals, y_vals = zip(*sorted_pairs)

        style = styles.get(case, {"color": "gray", "marker": "x"})
        label_name = legend_labels.get(case, case.capitalize())

        ax.scatter(
            x_vals,
            y_vals,
            label=label_name,
            color=style["color"],
            marker=style["marker"],
            s=60,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    if expected_error is not None:
        x_exp, y_exp = expected_error
        ax.plot(
            x_exp,
            y_exp,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Erro esperado",
            zorder=2,
        )

    ax.set_xscale(x_scale)
    ax.set_yscale("log")

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Erro relativo da permeabilidade (%)", fontsize=12, fontweight="bold"
    )

    ax.grid(True, which="major", linestyle="-", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    ax.legend(fontsize=10, framealpha=1.0, edgecolor="black")

    plt.tight_layout()

    os.makedirs(parent_folder_path, exist_ok=True)
    save_path = os.path.join(
        parent_folder_path, f"{simulation_name}_permeability_error.png"
    )
    plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def export_comparative_results_txt(
    results: dict,
    analytical_data: dict | None,
    experimental_data: dict | None,
    simulation_name: str,
    output_path: str,
) -> None:
    """Exports a wide-format comparative text report of the simulations.

    Creates an aligned, fixed-width text table comparing permeabilities (in mD)
    and errors across all theoretical references and LBM models.

    Args:
        results: Nested dict containing the computed simulation errors.
        analytical_data: Nested dict containing 3D and 2.5D theoretical data.
        experimental_data: Dict containing experimental permeabilities.
        simulation_name: Name of the simulation for the header.
        output_path: The full file path to save the .txt report.
    """
    um_to_md = 0.0009869233
    cases = ["fullscale", "greyscale", "2D", "laleian2015"]

    identifiers = set()
    for runs in results.values():
        identifiers.update(runs.keys())

    def get_numeric_ar(ident):
        match = re.search(r"(\d+(?:p\d+)?)", ident)
        return float(match.group(1).replace("p", ".")) if match else 0.0

    sorted_identifiers = sorted(list(identifiers), key=get_numeric_ar)

    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    divider = "-" * 230 + "\n"

    header = (
        f"{divider}"
        f" SIMULAÇÃO: {simulation_name} | DATA: {current_time}\n"
        f"{divider}"
    )

    col_width = 18
    table_header = f"{'Tag':<10} | "

    table_header += f"{'Exp [mD]':<{col_width}} | "
    table_header += f"{'Ana 3D [mD]':<{col_width}} | "
    table_header += f"{'Ana 2.5D [mD]':<{col_width}} | "
    for case in cases:
        table_header += f"{case.capitalize()} [mD] | ".ljust(col_width + 3)

    for case in cases:
        table_header += f"Err K {case.capitalize()} [%] | ".ljust(col_width + 3)

    for case in cases:
        table_header += f"Err V {case.capitalize()} | ".ljust(col_width + 3)

    table_header += "\n" + divider

    lines = []
    for ident in sorted_identifiers:
        match = re.search(r"(\d+(?:p\d+)?)", ident)
        tag = match.group(1) if match else ident

        line_str = f"{tag:<10} | "

        exp_k = (
            experimental_data.get(ident, float("nan"))
            if experimental_data
            else float("nan")
        )
        ana3d_k = (
            analytical_data["3d"][ident]["permeability"]
            if analytical_data
            and "3d" in analytical_data
            and ident in analytical_data["3d"]
            else float("nan")
        )
        ana25d_k = (
            analytical_data["2p5d"][ident]["permeability"]
            if analytical_data
            and "2p5d" in analytical_data
            and ident in analytical_data["2p5d"]
            else float("nan")
        )

        refs = [exp_k, ana3d_k, ana25d_k]
        for val in refs:
            val_md = val / um_to_md if val == val else float("nan")
            line_str += (
                f"{val_md:<{col_width}.2f} | "
                if val == val
                else f"{'NaN':<{col_width}} | "
            )

        for case in cases:
            perm = (
                results.get(case, {})
                .get(ident, {})
                .get("permeability", float("nan"))
            )
            perm_md = perm / um_to_md if perm == perm else float("nan")
            line_str += (
                f"{perm_md:<{col_width}.2f} | "
                if perm == perm
                else f"{'NaN':<{col_width}} | "
            )

        for case in cases:
            err_k = (
                results.get(case, {})
                .get(ident, {})
                .get("permeability_error", float("nan"))
            )
            line_str += (
                f"{err_k:<{col_width}.2f} | "
                if err_k == err_k
                else f"{'NaN':<{col_width}} | "
            )

        for case in cases:
            err_v = (
                results.get(case, {})
                .get(ident, {})
                .get("normalized_rmse", float("nan"))
            )
            line_str += (
                f"{err_v:<{col_width}.6f} | "
                if err_v == err_v
                else f"{'NaN':<{col_width}} | "
            )

        lines.append(line_str + "\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(header)
        file.write(table_header)
        file.writelines(lines)
        file.write(divider)
