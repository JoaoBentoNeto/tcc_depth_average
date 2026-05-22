import os
import sys
from typing import Literal, Sequence

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import results_analysis as analysis
import Simulation_Cases.analytical_equations as analytical
import workspace_generator as generate


def generate_aspect_ratios(
    aspect_ratios: Sequence[float],
    lower_bound: float = 1e-3,
    upper_bound: float = 1e3,
    num_points: int = 20,
    scale: Literal["log", "linear"] = "log",
) -> np.ndarray:
    """Generates an array of aspect ratios interpolating between fixed values.

    Creates an array containing the specified boundaries, the provided fixed
    aspect ratio values, and a specified number of interpolated points between
    each consecutive pair.

    Args:
        aspect_ratios: A sequence of fixed aspect ratio values to include.
        lower_bound: The lower limit of the domain. Defaults to 1e-3.
        upper_bound: The upper limit of the domain. Defaults to 1e3.
        num_points: The number of points to insert between each consecutive
            pair of fixed values. Defaults to 20.
        scale: The scaling method for interpolation, either 'log' or 'linear'.
            Defaults to 'log'.

    Returns:
        A NumPy array of generated aspect ratio values, sorted and without
        duplicates.

    Raises:
        ValueError: If 'scale' is 'log' and any of the points are less than
            or equal to zero.
    """
    fixed_points = sorted(set([lower_bound] + list(aspect_ratios) + [upper_bound]))
    generated_points = []

    for i in range(len(fixed_points) - 1):
        a = fixed_points[i]
        b = fixed_points[i + 1]

        if scale == "log":
            if a <= 0 or b <= 0:
                raise ValueError("For a 'log' scale, all points must be strictly positive.")
            interp = np.logspace(np.log10(a), np.log10(b), num_points + 2)
        else:
            interp = np.linspace(a, b, num_points + 2)

        generated_points.extend(interp[:-1])

    generated_points.append(fixed_points[-1])

    return np.array(generated_points)


if __name__ == "__main__":
    simulation_name = "duct"

    # win local: r"Z:\TCC\Part_1-literature_validation"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join("/home/bento/remote/hal", "TCC", "Part_1-literature_validation", simulation_name)
    remote_path = f"/home/joao.neto/TCC/Part_1-literature_validation/{simulation_name}"

    depth = 100
    lattice_length = 4
    resolution = 1.0

    tau = 0.9330127
    body_force = 1e-8

    aspect_ratios = [
        0.01,
        0.1,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2,
        3,
        5,
        7.5,
        10,
    ]
    # aspect_ratios = [1.0]

    simulation_data = analysis.extract_simulation_data(parent_folder_path=local_path)

    analytical_data = {"3d": {}, "2p5d": {}}

    for aspect_ratio in aspect_ratios:
        identifier = generate.format_identifier(value=aspect_ratio, symbol="AR")
        width = round((depth / resolution) / aspect_ratio) * resolution

        x_phys = np.linspace(-width / 2 + resolution / 2, width / 2 - resolution / 2, int(width / resolution))
        # print(f"x phys shape 1 {x_phys.shape}")

        k_3d = analytical.calculate_permeability_3d_rectangular_duct(width=width, height=depth) * width / (width + 2)
        k_2p5d = (
            analytical.calculate_permeability_2p5d_rectangular_duct(width=width, height=depth) * width / (width + 2)
        )

        u_3d = analytical.calculate_analytical_velocity_map_3d_rectangular_duct(
            x_coords=x_phys,
            height=depth,
            width=width,
            body_force=body_force,
            tau=tau,
            length_nodes=lattice_length,
            flow_axis=1,
        )
        # print(f"x phys shape 2 {x_phys.shape}")
        u_2p5d = analytical.calculate_analytical_velocity_map_2p5d_rectangular_duct(
            x_coords=x_phys,
            height=depth,
            width=width,
            body_force=body_force,
            tau=tau,
            length_nodes=lattice_length,
            flow_axis=1,
        )
        # print(f"u_2p5d (ar = {aspect_ratio}) = {u_2p5d.shape}")

        analytical_data["3d"][identifier] = {
            "permeability": k_3d,
            "velocity": u_3d,
        }

        analytical_data["2p5d"][identifier] = {
            "permeability": k_2p5d,
            "velocity": u_2p5d,
        }

    errors_data = analysis.compute_errors(
        simulation_data=simulation_data,
        analytical_data=analytical_data,
        experimental_data=None,
    )

    analysis.export_comparative_results_txt(
        results=errors_data,
        analytical_data=analytical_data,
        experimental_data=None,
        simulation_name=simulation_name,
        output_path=os.path.join(local_path, f"{simulation_name}_comparative_results.txt"),
    )

    expected_x = []
    expected_y = []

    n_aprox = int(1000 / (len(aspect_ratios) + 1)) - 1
    razao_aspecto = generate_aspect_ratios(
        aspect_ratios=aspect_ratios,
        lower_bound=1e-3,
        upper_bound=1e3,
        num_points=1000,
        scale="log",
    )

    for aspect_ratio in razao_aspecto:
        width = depth / aspect_ratio
        k_3d = analytical.calculate_permeability_3d_rectangular_duct(width=width, height=depth) * width / (width + 2)
        k_2p5d = (
            analytical.calculate_permeability_2p5d_rectangular_duct(width=width, height=depth) * width / (width + 2)
        )
        expected_x.append(aspect_ratio)
        expected_y.append(
            analysis.calculate_permeability_errors(
                test_permeability=k_2p5d,
                ref_permeability=k_3d,
            )
        )

    analysis.plot_permeability_errors(
        parent_folder_path=local_path,
        simulation_name=simulation_name,
        results=errors_data,
        x_label="Razão de Aspecto (AR = h/b)",
        x_scale="log",
        expected_error=(expected_x, expected_y),
    )
