import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import results_analysis as analysis

if __name__ == "__main__":
    simulation_name = "unit_cell"

    # win local: r"Z:\TCC\Part_2-semi_width_obtaining"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join(
        "/home/bento/remote/hal",
        "TCC",
        "Part_2-semi_width_obtaining",
        simulation_name,
    )
    remote_path = f"/home/joao.neto/TCC/Part_2-semi_width_obtaining/{simulation_name}"
    local_path = remote_path
    simulation_data = analysis.extract_simulation_data(parent_folder_path=local_path)

    errors_data = analysis.compute_errors(
        simulation_data=simulation_data,
        analytical_data=None,
        experimental_data=None,
    )

    analysis.export_comparative_results_txt(
        results=errors_data,
        analytical_data=None,
        experimental_data=None,
        simulation_name=simulation_name,
        output_path=os.path.join(
            local_path, f"{simulation_name}_comparative_results.txt"
        ),
    )

    expected_x = np.linspace(0.1, 1.4, 1000)
    expected_y = np.ones_like(expected_x) * 0.8

    analysis.plot_permeability_errors(
        parent_folder_path=local_path,
        simulation_name=simulation_name,
        results=errors_data,
        title="Célula Unitária",
        x_label="Tamanho do Voxel [\u03bcm/voxel]",
        x_scale="linear",
        expected_error=(expected_x, expected_y),
    )
