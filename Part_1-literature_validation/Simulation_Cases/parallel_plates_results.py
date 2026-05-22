import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import results_analysis as analysis
import Simulation_Cases.analytical_equations as analytical
import workspace_generator as generate

if __name__ == "__main__":
    simulation_name = "parallel_plates"

    # win local: r"Z:\TCC\Part_1-literature_validation"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join("/home/bento/remote/hal", "TCC", "Part_1-literature_validation", simulation_name)
    remote_path = f"/home/joao.neto/TCC/Part_1-literature_validation/{simulation_name}"

    lattice_length = 4
    lattice_width = 16

    tau = 0.9330127
    body_force = 1e-8

    timestepmax = 1000000
    tolerance = 1e-12

    depths_resolution = [
        (0.1, 0.01),
        (1, 0.1),
        (5, 0.5),
        (10, 1.0),
        (20, 2.0),
        (30, 3.0),
        (50, 5.0),
        (75, 7.5),
        (100, 10.0),
        (150, 15.0),
        (200, 20.0),
    ]

    simulation_data = analysis.extract_simulation_data(parent_folder_path=local_path)

    analytical_data = {"3d": {}}

    for depth, resolution in depths_resolution:
        identifier = generate.format_identifier(value=depth, symbol="h")

        k = analytical.calculate_permeability_parallel_plates(height=depth)

        u = analytical.calculate_analytical_velocity_map_parallel_plates(
            length_nodes=lattice_length,
            width_nodes=lattice_width,
            height_nodes=round(depth / resolution),
            body_force=body_force,
            tau=tau,
        )

        analytical_data["3d"][identifier] = {
            "permeability": k,
            "velocity": u,
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

    analysis.plot_permeability_errors(
        parent_folder_path=local_path,
        simulation_name=simulation_name,
        results=errors_data,
        x_label="Profundidade (h)",
        x_scale="log",
        expected_error=None,
    )
