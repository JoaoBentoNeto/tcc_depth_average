import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import geometries
import numpy as np
import workspace_generator as generate

if __name__ == "__main__":
    hardware_type = "a100"

    simulation_name = "double_pore"

    # win local: r"Z:\TCC\Part_3-rectangular_duct_model_implementation"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join(
        "/home/bento/remote/hal",
        "TCC",
        "Part_3-rectangular_duct_model_implementation",
        simulation_name,
    )
    remote_path = f"/home/joao.neto/TCC/Part_3-rectangular_duct_model_implementation/{simulation_name}"
    local_path = remote_path
    generate.write_sbatch_header(local_parent_path=local_path)

    tau = 0.9330127
    body_force = 1e-8

    resolution = 0.5

    timestepmax = 1000000
    tolerance = 1e-10

    simulation_geometry = geometries.create_double_pore(
        eccentricity=(22, 22), pore_rx=270, pore_ry=150, obstacle_rx=150, obstacle_ry=90
    )

    depths = np.round(
        geometries.calculate_simulation_depths(geometry_mask=simulation_geometry)
        * resolution
    )

    for depth in depths:
        identifier = generate.format_identifier(value=depth, symbol="h")
        depth_map = simulation_geometry * depth

        # for model in ["rdm", "laleian2015"]:
        #     generate.depth_averaged_workspace(
        #         tau=tau,
        #         body_force=body_force,
        #         depth_map=depth_map,
        #         local_path=local_path,
        #         remote_path=remote_path,
        #         simulation_name=simulation_name + f"_{model}{identifier}",
        #         resolution=resolution,
        #         timestepmax=timestepmax,
        #         tolerance=tolerance,
        #         model= model,
        #     )
        # print(f"Created double pore python {model} h = {depth}")

        generate.fullscale_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_full{identifier}",
            resolution=resolution,
            hardware_type=hardware_type,
            timestepmax=timestepmax,
            tolerance=tolerance,
            only_one_domain=False,
        )
        print(f"Created double pore fullscale h = {depth}")

        generate.bidimensional_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_2d{identifier}",
            resolution=resolution,
            hardware_type=hardware_type,
            timestepmax=timestepmax,
            tolerance=tolerance,
            only_one_domain=False,
        )
        print(f"Created double pore 2d h = {depth}")

        for model in ["rdm", "laleian2015"]:
            generate.greyscale_workspace(
                tau=tau,
                body_force=body_force,
                depth_map=depth_map,
                local_path=local_path,
                remote_path=remote_path,
                simulation_name=simulation_name + f"_lbpm_{model}{identifier}",
                model=model,
                resolution=resolution,
                hardware_type=hardware_type,
                timestepmax=timestepmax,
                tolerance=tolerance,
                only_one_domain=False,
            )
            print(f"Created double pore lbpm {model} h = {depth}")
