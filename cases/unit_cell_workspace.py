import os

import geometries as geo
import workspace_generator as wg

if __name__ == "__main__":
    hardware_type = "cpu"

    simulation_name = "unit_cell"

    # win local: r"Z:\TCC"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join("/home/bento/remote/hal", "TCC", simulation_name)
    remote_path = f"/home/joao.neto/TCC/{simulation_name}"
    wg.write_sbatch_header(local_parent_path=local_path)

    resolution = 1.25
    radius = 335 / 4

    # depths = [
    #     round(radius / 8),
    #     round(radius / 6),
    #     round(radius / 4),
    #     round(radius / 2),
    #     round(radius / 1.5),
    #     radius,
    #     round(radius * 1.5),
    #     2 * radius,
    #     4 * radius,
    # ]

    depths = [20]
    tau = 1.1

    dx = 1.25e-6
    nu_phy = 1e-6
    g_phy = 9.8e-2

    nu_lb = 1 / 3 * (tau - 0.5)
    dt = nu_lb / nu_phy * dx**2
    body_force = g_phy * (dt * 1 / 2) ** 2 / dx

    relative_distance = 2
    horizontal_repeats = 1
    vertical_repeats = 1

    timestepmax = 1000000
    tolerance = 1e-12

    for depth in depths:
        identifier = wg.format_identifier(value=depth, symbol="h")

        depth_map = (
            geo.create_unit_cell(
                radius=float(round(radius / resolution)),
                relative_distance=relative_distance,
                vertical_repeats=vertical_repeats,
                horizontal_repeats=horizontal_repeats,
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
