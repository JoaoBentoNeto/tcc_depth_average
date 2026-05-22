import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import geometries
import workspace_generator as generate

if __name__ == "__main__":
    hardware_type = "k40m"

    simulation_name = "duct"

    # win local: r"Z:\TCC\Part_2-rectangular_duct_model_implementation"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join(
        "/home/bento/remote/hal", "TCC", "Part_2-rectangular_duct_model_implementation", simulation_name
    )
    remote_path = f"/home/joao.neto/TCC/Part_2-rectangular_duct_model_implementation/{simulation_name}"
    # local_path = remote_path

    generate.write_sbatch_header(local_parent_path=local_path)

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
    ]
    # aspect_ratios = [1]

    for aspect_ratio in aspect_ratios[::-1]:
        identifier = generate.format_identifier(value=aspect_ratio, symbol="AR")
        depth_map = (
            geometries.create_duct(
                aspect_ratio=aspect_ratio,
                height=depth,
                resolution=resolution,
                lattice_length=lattice_length,
            )
            * depth
        )

        generate.depth_average_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_laleian2015{identifier}",
            simulation_type="laleian2015",
            resolution=resolution,
            timestepmax=timestepmax,
            tolerance=tolerance,
        )

        generate.depth_average_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_rdm{identifier}",
            simulation_type="rdm",
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

        print(f"Generated AR={aspect_ratio} simulation case")
