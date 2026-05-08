import os

import geometries as geo
import workspace_generator as wg

if __name__ == "__main__":
    gpu_type = "k40m"

    simulation_name = "duct"

    # win local: r"Z:\TCC"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join("/home/bento/remote/hal", "TCC", simulation_name)
    remote_path = f"/home/joao.neto/TCC/{simulation_name}"
    wg.write_sbatch_header(local_parent_path=local_path)

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

    for aspect_ratio in aspect_ratios:
        identifier = wg.format_identifier(value=aspect_ratio, symbol="AR")
        depth_map = geo.create_duct_2d(
            aspect_ratio=aspect_ratio,
            height=depth,
            resolution=resolution,
            lattice_length=lattice_length,
        )

        wg.laleian2015_workspace(
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

        wg.fullscale_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_full{identifier}",
            resolution=resolution,
            gpu_type=gpu_type,
            timestepmax=timestepmax,
            tolerance=tolerance,
            only_one_domain=False,
        )

        wg.greyscale_workspace(
            tau=tau,
            body_force=body_force,
            depth_map=depth_map,
            local_path=local_path,
            remote_path=remote_path,
            simulation_name=simulation_name + f"_grey{identifier}",
            resolution=resolution,
            gpu_type=gpu_type,
            timestepmax=timestepmax,
            tolerance=tolerance,
            only_one_domain=False,
        )
