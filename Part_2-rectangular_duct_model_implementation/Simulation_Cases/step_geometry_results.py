import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import results_analysis as analysis

if __name__ == "__main__":
    simulation_name = "step"

    # win local: r"Z:\TCC\Part_2-rectangular_duct_model_implementation"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join(
        "/home/bento/remote/hal", "TCC", "Part_2-rectangular_duct_model_implementation", simulation_name
    )
    remote_path = f"/home/joao.neto/TCC/Part_2-rectangular_duct_model_implementation/{simulation_name}"

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
