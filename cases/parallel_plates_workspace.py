import os

import geometries as geo
import workspace_generator as wg

if __name__ == "__main__":
    hardware_type = "cpu"

    simulation_name = "parallel_plates"

    # win local: r"Z:\TCC"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join("/home/bento/remote/hal", "TCC", simulation_name)
    remote_path = f"/home/joao.neto/TCC/{simulation_name}"
    wg.write_sbatch_header(local_parent_path=local_path)

    lattice_length = 4
    lattice_width = 32
    resolution = 1.0

    tau = 0.9330127
    body_force = 1e-8

    timestepmax = 1000000
    tolerance = 1e-12

    depths = [
        10,
        20,
        50,
        75,
        100,
    ]

    for depth in depths:
        identifier = wg.format_identifier(value=depth, symbol="h")
        depth_map = (
            geo.create_parallel_plates(
                lattice_width=lattice_width,
                lattice_length=lattice_length,
            )
            * depth
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
            hardware_type=hardware_type,
            timestepmax=timestepmax,
            tolerance=tolerance,
            only_one_domain=False,
        )

        wg.bidimensional_workspace(
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

        wg.greyscale_workspace(
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
