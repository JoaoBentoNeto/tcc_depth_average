import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import geometries
import workspace_generator as generate

if __name__ == "__main__":
    hardware_type = "cpu"

    simulation_name = "parallel_plates"

    # win local: r"Z:\TCC\Part_1-literature_validation"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join("/home/bento/remote/hal", "TCC", "Part_1-literature_validation", simulation_name)
    remote_path = f"/home/joao.neto/TCC/Part_1-literature_validation/{simulation_name}"
    generate.write_sbatch_header(local_parent_path=local_path)

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

    for depth, resolution in depths_resolution:
        identifier = generate.format_identifier(value=depth, symbol="h")
        depth_map = (
            geometries.create_parallel_plates(
                lattice_width=lattice_width,
                lattice_length=lattice_length,
            )
            * depth
        )

        generate.laleian2015_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_laleian2015{identifier}",
            resolution=resolution,
            timestepmax=timestepmax,
            tolerance=tolerance,
        )

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

        generate.greyscale_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_grey{identifier}",
            resolution=resolution,
            hardware_type=hardware_type,
            timestepmax=timestepmax,
            tolerance=tolerance,
            only_one_domain=False,
        )
