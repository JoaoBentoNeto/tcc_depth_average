import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import geometries
import workspace_generator as generate

if __name__ == "__main__":
    hardware_type = "k40m"

    simulation_name = "unit_cell"

    # win local: r"Z:\TCC\Part_2-rectangular_duct_model_implementation"
    # linux local: "/home/bento/remote/hal"
    local_path = os.path.join(
        "/home/bento/remote/hal", "TCC", "Part_2-rectangular_duct_model_implementation", simulation_name
    )
    remote_path = f"/home/joao.neto/TCC/Part_2-rectangular_duct_model_implementation/{simulation_name}"
    generate.write_sbatch_header(local_parent_path=local_path)

    radius = 335 / 4

    depth = 20
    tau = 1.1

    resolutions = [0.25, 0.5, 0.75, 1.0, 1.25]

    relative_distance = 2
    horizontal_repeats = 1
    vertical_repeats = 1

    timestepmax = 1000000
    tolerance = 1e-12

    nu_phy = 1e-6
    g_phy = 9.8e-2

    nu_lb = 1 / 3 * (tau - 0.5)

    for resolution in resolutions:
        identifier = generate.format_identifier(value=resolution, symbol="dx")

        dx = resolution * 1e-6
        dt = nu_lb / nu_phy * dx**2
        body_force = g_phy * (dt * 1 / 2) ** 2 / dx

        depth_map = (
            geometries.create_unit_cell(
                radius=float(round(radius / resolution)),
                relative_distance=relative_distance,
                vertical_repeats=vertical_repeats,
                horizontal_repeats=horizontal_repeats,
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
            simulation_name=simulation_name + f"rdm{identifier}",
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
