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

    radius = 335 / 4

    depth = 20
    tau = 1.1

    resolutions = [0.5, 0.75, 1.0, 1.25]

    relative_distance = 2
    horizontal_repeats = 1
    vertical_repeats = 1

    timestepmax = 1000000
    tolerance = 1e-12

    nu_phy = 1e-6
    g_phy = 9.8e-2

    nu_lb = 1 / 3 * (tau - 0.5)

    obtaining_methods = [
        "skeletonize",
        "medial_axis",
        "n_edt",
        "local_thickness",
        "laleian2015",
    ]

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
