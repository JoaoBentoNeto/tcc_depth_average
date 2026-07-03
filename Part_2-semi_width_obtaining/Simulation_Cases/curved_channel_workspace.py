import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import geometries
import workspace_generator as generate

if __name__ == "__main__":
    hardware_type = "cpu"

    simulation_name = "curved_channel"

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

    generate.write_sbatch_header(local_parent_path=local_path)

    tau = 0.9330127
    body_force = 1e-8

    resolution = 0.5

    timestepmax = 1000000
    tolerance = 1e-12

    simulation_geometry = geometries.create_curved_channel(domain_size=300)

    depths = np.round(
        geometries.calculate_simulation_depths(geometry_mask=simulation_geometry)
        * resolution
    )

    obtaining_methods = [
        "skeletonize",
        "medial_axis",
        "n_edt",
        "local_thickness",
        "laleian2015",
        "perfect",
    ]

    for depth in depths:
        identifier = generate.format_identifier(value=depth, symbol="h")
        print(f"generating case {identifier}")

        depth_map = simulation_geometry * depth

        generate.fullscale_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_full{identifier}",
            resolution=resolution,
            hardware_type="k40m" if hardware_type == "cpu" else hardware_type,
            timestepmax=timestepmax,
            tolerance=tolerance,
            only_one_domain=False,
        )

        for obtain_method in obtaining_methods:
            generate.greyscale_workspace(
                tau=tau,
                body_force=body_force,
                depth_map=depth_map,
                local_path=local_path,
                remote_path=remote_path,
                simulation_name=simulation_name + f"_grey_{obtain_method}{identifier}",
                obtaining_method=obtain_method,
                resolution=resolution,
                hardware_type=hardware_type,
                timestepmax=timestepmax,
                tolerance=tolerance,
                only_one_domain=False,
            )
