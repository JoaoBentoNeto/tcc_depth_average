import os

import numpy as np

import cases.rectangular_duct_analytical_equations as analytical
import results_analysis as ra
import workspace_generator as wg


from typing import Literal, Sequence
import numpy as np


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
    fixed_points = sorted(
        set([lower_bound] + list(aspect_ratios) + [upper_bound])
    )
    generated_points = []

    for i in range(len(fixed_points) - 1):
        a = fixed_points[i]
        b = fixed_points[i + 1]

        if scale == "log":
            if a <= 0 or b <= 0:
                raise ValueError(
                    "For a 'log' scale, all points must be strictly positive."
                )
            interp = np.logspace(np.log10(a), np.log10(b), num_points + 2)
        else:
            interp = np.linspace(a, b, num_points + 2)

        generated_points.extend(interp[:-1])

    generated_points.append(fixed_points[-1])

    return np.array(generated_points)


if __name__ == "__main__":
    simulation_name = "duct"

    # win local: r"Z:\TCC"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join("/home/bento/remote/hal", "TCC", simulation_name)
    remote_path = f"/home/joao.neto/TCC/{simulation_name}"

    depth = 100
    lattice_length = 4
    resolution = 1.0

    tau = 0.9330127
    body_force = 1e-8

    timestepmax = 1000000
    tolerance = 1e-12

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
        20,
    ]
    # aspect_ratios = [1.0]

    analytical_data = {"3d": {}, "2p5d": {}}

    for aspect_ratio in aspect_ratios:
        identifier = wg.format_identifier(value=aspect_ratio, symbol="AR")
        width = round((depth / resolution) / aspect_ratio) * resolution

        x_phys = np.linspace(
            resolution / 2, width - resolution / 2, int(width / resolution)
        )
        y_phys = np.linspace(
            resolution / 2,
            lattice_length * resolution - resolution / 2,
            lattice_length,
        )
        xx, yy = np.meshgrid(x_phys, y_phys, indexing="ij")

        k_3d = analytical.calculate_permeability_3d(width=width, height=depth)
        k_2p5d = analytical.calculate_permeability_2p5d(
            width=width, height=depth
        )

        u_3d = analytical.calculate_analytical_velocity_map_3d(
            x_coords=x_phys,
            height=depth,
            width=width,
            body_force=body_force,
            tau=tau,
            length_nodes=lattice_length,
            flow_axis=1,
        )
        u_2p5d = analytical.calculate_analytical_velocity_map_2p5d(
            x_coords=x_phys,
            height=depth,
            width=width,
            body_force=body_force,
            tau=tau,
            length_nodes=lattice_length,
            flow_axis=1,
        )

        analytical_data["3d"][identifier] = {
            "permeability": k_3d,
            "velocity": u_3d,
            "coords": (xx, yy),
        }

        analytical_data["2p5d"][identifier] = {
            "permeability": k_2p5d,
            "velocity": u_2p5d,
            "coords": (xx, yy),
        }

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
        k_3d = analytical.calculate_permeability_3d(width=width, height=depth)
        k_2p5d = analytical.calculate_permeability_2p5d(
            width=width, height=depth
        )
        expected_x.append(aspect_ratio)
        expected_y.append(
            ra.calculate_permeability_errors(
                test_permeability=k_2p5d,
                ref_permeability=k_3d,
            )
        )

    simulation_data = ra.extract_simulation_data(parent_folder_path=local_path)

    errors_data = ra.compute_errors(
        simulation_data=simulation_data,
        analytical_data=analytical_data,
        experimental_data=None,
    )

    ra.plot_permeability_errors(
        parent_folder_path=local_path,
        simulation_name=simulation_name,
        results=errors_data,
        x_label="Razão de Aspecto (AR = h/b)",
        x_scale="log",
        expected_error=(expected_x, expected_y),
    )

    ra.export_comparative_results_txt(
        results=errors_data,
        analytical_data=analytical_data,
        experimental_data=None,
        simulation_name=simulation_name,
        output_path=local_path,
    )
