import os  # noqa: D100
import posixpath
import textwrap
from datetime import datetime
from typing import Literal

import numpy as np


def create_simulation_directory(
    simulation_name: str,
    remote_parent_path: str,
    local_parent_path: str,
    simulation_type: Literal["full", "2D", "grey", "laleian2015"],
) -> tuple[str, str]:
    """Creates a local directory and generates local and remote paths.

    Args:
        simulation_name: The base name of the directory to be created.
        remote_parent_path: The parent path on the Linux cluster.
        local_parent_path: The local parent directory.
        simulation_type: Defines the routine and environment to be used.
            Must be exactly "full", "grey", or "laleian2015".

    Returns:
        A tuple containing two strings:
        - local_path: The absolute path to the created directory.
        - remote_path: The corresponding path formatted for Linux.
    """
    if simulation_type == "full":
        local_path = os.path.join(local_parent_path, "fullscale", simulation_name)
        remote_path = posixpath.join(remote_parent_path, "fullscale", simulation_name)
    if simulation_type == "2D":
        local_path = os.path.join(local_parent_path, "2d", simulation_name)
        remote_path = posixpath.join(remote_parent_path, "2d", simulation_name)
    if simulation_type == "grey":
        local_path = os.path.join(local_parent_path, "greyscale", simulation_name)
        remote_path = posixpath.join(remote_parent_path, "greyscale", simulation_name)
    if simulation_type == "laleian2015":
        local_path = os.path.join(local_parent_path, "laleian2015", simulation_name)
        remote_path = posixpath.join(remote_parent_path, "laleian2015", simulation_name)

    os.makedirs(local_path, exist_ok=True)

    return local_path, remote_path


def divide_domain(
    domain: tuple[int, int, int],
    hardware_type: Literal["k40m", "a100", "cpu"],
    only_one_domain: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Finds the best geometric division to maximize hardware usage.

    Args:
        domain: A tuple containing the total volume dimensions (nx, ny, nz).
        hardware_type: The identifier for the hardware architecture.
            Must be exactly "k40m", "a100" or "cpu".
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        A tuple containing two tuples:
        - The first tuple is the processor grid layout (procs_x, procs_y, procs_z).
        - The second tuple is the subdomain dimensions (sub_nx, sub_ny, sub_nz).
    """
    nx, ny, nz = domain

    if only_one_domain:
        return (1, 1, 1), (nx, ny, nz)

    max_procs = 16 if hardware_type == "cpu" else 4

    possible_values = list(range(max_procs, 0, -1))

    for num_procs in possible_values:
        best_division = None
        low_score = float("inf")

        for n_procs_x in range(1, num_procs + 1):
            for n_procs_y in range(1, (num_procs // n_procs_x) + 1):
                n_procs_z = num_procs // (n_procs_x * n_procs_y)

                if n_procs_x * n_procs_y * n_procs_z == num_procs:
                    if nx % n_procs_x == 0 and ny % n_procs_y == 0 and nz % n_procs_z == 0:
                        subdomain_x = nx // n_procs_x
                        subdomain_y = ny // n_procs_y
                        subdomain_z = nz // n_procs_z

                        score = subdomain_x + subdomain_y + subdomain_z

                        if score < low_score:
                            low_score = score
                            best_division = (
                                (n_procs_x, n_procs_y, n_procs_z),
                                (subdomain_x, subdomain_y, subdomain_z),
                            )

        if best_division:
            return best_division

    return (1, 1, 1), (nx, ny, nz)


def create_slurm_script(
    output_path: str,
    simulation_name: str,
    domain: tuple[int, int, int],
    hardware_type: Literal["k40m", "a100", "cpu"],
    simulation_type: Literal["full", "2D", "grey", "laleian2015"],
    only_one_domain: bool = False,
) -> None:
    """Generates a SLURM batch script.

    Args:
        output_path: Directory path where the .sh script will be saved.
        simulation_name: Name of the simulation.
        domain: A tuple containing the total volume dimensions (nx, ny, nz).
        hardware_type: The identifier for the hardware architecture.
            Must be exactly "k40m", "a100" or "cpu".
        simulation_type: Defines the routine and environment to be used.
            Must be exactly "full","2D","grey", or "laleian2015".
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        None. The script is written directly to the file system.
    """
    script_path = os.path.join(output_path, f"{simulation_name}.sh")

    header = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={simulation_name}
        #SBATCH --output=output_%j.log
        #SBATCH --error=error_%j.log
        #SBATCH --nodes=1
        #SBATCH --time=00:00:00
        """)

    if simulation_type == "laleian2015":
        body = textwrap.dedent(f"""\
            #SBATCH --partition=close_cpu
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=16

            module load conda
            conda activate depth_averaged
            python -u /home/joao.neto/TCC/Part_1-literature_validation/run_laleian2015.py ./{simulation_name}.db
            """)
    else:
        n_procs, _ = divide_domain(
            domain=domain,
            hardware_type=hardware_type,
            only_one_domain=only_one_domain,
        )
        total_processors = int(n_procs[0] * n_procs[1] * n_procs[2])

        if simulation_type in ("full", "2D"):
            routine = "lbpm_permeability_simulator"
            module = "lbpm/cpu/dev" if hardware_type == "cpu" else "lbpm/gpu"

        elif simulation_type == "grey":
            routine = "lbpm_greyscale_simulator"
            module = "lbpm/cpu" if hardware_type == "cpu" else "lbpm/gpu"
            module = module + "/poro_grey_fix_2edb227"

        else:
            raise ValueError(f"Invalid simulation_type: {simulation_type}")

        if hardware_type == "cpu":
            hardware_directives = textwrap.dedent(f"""\
                #SBATCH --partition=close_cpu
                #SBATCH --ntasks={total_processors}
                #SBATCH --cpus-per-task=1
                """)
        else:
            hardware_directives = textwrap.dedent(f"""\
                #SBATCH --partition=all_gpu
                #SBATCH --ntasks={total_processors}
                #SBATCH --gres=gpu:{hardware_type}:{total_processors}
                #SBATCH --cpus-per-task=1
                """)
        if hardware_type == "cpu" and simulation_type == "grey":
            hardware_directives = hardware_directives + textwrap.dedent("""\

                module use "$HOME/modulefiles"
                """)

        body = hardware_directives + textwrap.dedent(f"""\


            module load {module}
            mpirun -np {total_processors} {routine} ./{simulation_name}.db
            """)

    with open(script_path, "w") as file:
        file.write(header + body)


def common_db(
    output_path: str,
    simulation_name: str,
    domain: tuple[int, int, int],
    resolution: float,
    component_labels: tuple[int, ...],
    hardware_type: Literal["k40m", "a100", "cpu"],
    only_one_domain: bool = False,
) -> None:
    """Generates a database script that is common for both fullscale and
    greyscale LBPM simulations.

    Args:
        output_path: Directory path where the .db script will be saved.
        simulation_name: Name of the simulation.
        domain: A tuple containing the total volume dimensions (nx, ny, nz).
        resolution: A number indicating the conversion in micrometers/voxel.
        component_labels: A tuple of integer labels representing the distinct
            phases in the domain (e.g., 0 for solid, others for fluid).
        hardware_type: The identifier for the hardware architecture.
            Must be exactly "k40m", "a100" or "cpu".
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        None. The script is written directly to the file system.
    """  # noqa: D205
    read_write_values = ", ".join(f"{label}" for label in component_labels)
    n_procs, subdomains = divide_domain(
        domain=domain,
        hardware_type=hardware_type,
        only_one_domain=only_one_domain,
    )

    content = textwrap.dedent(f"""\
        Domain {{
            Filename = "{simulation_name}.raw"
            ReadType = "8bit"
            offset = 0, 0, 0
            voxel_length = {resolution}
            N = {domain[2]}, {domain[1]}, {domain[0]}
            nproc = {n_procs[2]}, {n_procs[1]}, {n_procs[0]}
            n = {subdomains[2]}, {subdomains[1]}, {subdomains[0]}
            ReadValues = {read_write_values}   
            WriteValues = {read_write_values}
            InletLayers = 0, 0, 0
            OutletLayers = 0, 0, 0
            BC = 0
        }}
        Visualization {{
            format = "vtk"
            write_silo = true
            save_8bit_raw = false
            save_phase_field = true
            save_pressure = true
            save_velocity = true
        }}""")

    with open(os.path.join(output_path, f"{simulation_name}.db"), "w") as file:
        file.write(content)


def greyscale_db(
    tau: float,
    body_force: float,
    output_path: str,
    simulation_name: str,
    timestepmax: int,
    tolerance: float,
) -> None:
    """Complements the database script for greyscale LBPM simulations.

    Args:
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        output_path: Directory path where the .db script is located.
        simulation_name: Name of the simulation.
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.

    Returns:
        None. The script appends the configuration directly to the file system.
    """
    content = textwrap.dedent(f"""\
        
        Greyscale {{
            tau = {tau}
            F = 0.0, 0.0, {body_force}
            timestepMax = {timestepmax}
            tolerance = {tolerance}
            collision = "MRT"
            FileVoxelPorosityMap = "porosity_map_{simulation_name}.raw", "double"
            FileVoxelPermeabilityMap = "permeability_map_{simulation_name}.raw", "double"
            Forchheimer = false
        }}
        Analysis {{
            analysis_interval = 1000
            subphase_analysis_interval = 5000
            visualization_interval = 1000000
            N_threads = 1
            restart_interval = 1000000
            restart_file = "Restart"
        }}""")

    with open(os.path.join(output_path, f"{simulation_name}.db"), "a") as file:
        file.write(content)


def fullscale_db(
    tau: float,
    body_force: float,
    output_path: str,
    simulation_name: str,
    timestepmax: int,
    tolerance: float,
) -> None:
    """Complements the database script for fullscale permeability LBPM
        simulations.

    Args:
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        output_path: Directory path where the .db script is located.
        simulation_name: Name of the simulation.
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.

    Returns:
        None. The script appends the configuration directly to the file system.
    """  # noqa: D205
    content = textwrap.dedent(f"""\
        
        MRT {{
            tau = {tau}
            F = 0.0, 0.0, {body_force}
            timestepMax = {timestepmax}
            tolerance = {tolerance}
        }}""")

    with open(os.path.join(output_path, f"{simulation_name}.db"), "a") as file:
        file.write(content)


def laleian2015_db(
    local_path: str,
    remote_path: str,
    simulation_name: str,
    domain: tuple[int, int],
    resolution: float,
    tau: float,
    body_force: float,
    timestepmax: int,
    tolerance: float,
    on_cluster: bool = True,
) -> None:
    """Generates the database script for custom depth-averaged LBM simulations.

    Args:
        local_path: Directory path where local archives are located. Also
            used as the destination to save the generated .db file.
        remote_path: Directory path for archives when using the cluster.
            Must be written in Linux format.
        simulation_name: Name of the simulation.
        domain: A tuple containing the 2D volume dimensions (nx, ny).
        resolution: A number indicating the conversion in micrometers/voxel.
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.
        on_cluster: If True, the internal config references the remote path.

    Returns:
        None. The script is written directly to the local file system.
    """
    simulation_path = remote_path if on_cluster else local_path

    content = textwrap.dedent(f"""\
        LBM Depth-averaged Laleian 2015 {{
            tau = {tau}
            F = {body_force}, 0.0
            timestepMax = {timestepmax}
            tolerance = {tolerance}
            simulation_path = "{simulation_path}"
            simulation_name = "{simulation_name}"
            voxel_length = {resolution}
            N = {domain[0]}, {domain[1]}
        }}""")

    with open(os.path.join(local_path, f"{simulation_name}.db"), "w") as file:
        file.write(content)


def write_sbatch_header(local_parent_path: str) -> None:
    """Writes a timestamped header to organize all sbatch master files.

    Creates the parent directory if it does not exist, then appends a
    timestamped visual separator to the master files for fullscale, 2D,
    greyscale, and laleian2015 simulations.

    Args:
        local_parent_path: The local directory where the .txt files are saved.

    Returns:
        None. The headers are written directly to the file system.
    """
    os.makedirs(local_parent_path, exist_ok=True)

    mapping = {
        "full": "sbatches_fullscale.txt",
        "grey": "sbatches_greyscale.txt",
        "laleian2015": "sbatches_laleian2015.txt",
        "2d": "sbatches_2d.txt",
        "all": "sbatches_all.txt",
    }

    for simulation_type in ["full", "2d", "grey", "laleian2015", "all"]:
        filename = mapping.get(simulation_type)
        txt_path = os.path.join(local_parent_path, filename)

        now = datetime.now().strftime("%d/%m/%Y %H:%M")

        header = textwrap.dedent(f"""\
            
            
            =========================================
            Data: {now}
            -----------------------------------------

            """)

        with open(txt_path, "a") as file:
            file.write(header)


def append_sbatch_command(
    local_parent_path: str,
    remote_simulation_path: str,
    simulation_name: str,
    simulation_type: Literal["full", "2D", "grey", "laleian2015"],
) -> None:
    """Appends the submission command to a type-specific sbatches text file.

    Args:
        local_parent_path: The local directory where the .txt will be saved.
        remote_simulation_path: The path of the simulation on the cluster.
        simulation_name: The name of the simulation job.
        simulation_type: Defines the target master file. Must be exactly
            "full", "2D", "grey", or "laleian2015".

    Raises:
        ValueError: If an invalid simulation_type is provided.

    Returns:
        None. The command is written directly to the file.
    """
    if simulation_type == "full":
        txt_filename = "sbatches_fullscale.txt"
    elif simulation_type == "grey":
        txt_filename = "sbatches_greyscale.txt"
    elif simulation_type == "laleian2015":
        txt_filename = "sbatches_laleian2015.txt"
    elif simulation_type == "2D":
        txt_filename = "sbatches_2d.txt"
    else:
        raise ValueError(f"Invalid simulation_type: {simulation_type}")

    txt_path = os.path.join(local_parent_path, txt_filename)
    all_path = os.path.join(local_parent_path, "sbatches_all.txt")

    command = f"cd {remote_simulation_path} && dos2unix {simulation_name}.sh && sbatch {simulation_name}.sh\n"

    with open(txt_path, "a") as file:
        file.write(command)
    with open(all_path, "a") as file:
        file.write(command)


def greyscale_workspace(
    tau: float,
    body_force: float,
    depth_map: np.ndarray,
    local_path: str,
    remote_path: str,
    simulation_name: str,
    resolution: float,
    hardware_type: Literal["k40m", "a100", "cpu"],
    timestepmax: int,
    tolerance: float,
    only_one_domain: bool = False,
) -> None:
    """Orchestrates the workspace creation for greyscale LBPM simulations.

    Args:
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        depth_map: A 2D NumPy array representing the depth distribution.
        local_path: The local parent directory to store the workspace.
        remote_path: The parent directory on the Linux cluster.
        simulation_name: Name of the simulation.
        resolution: Conversion factor in micrometers/voxel.
        hardware_type: The identifier for the hardware architecture .
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        None. The script creates the workspace directly in the file system.
    """
    local_sim_path, remote_sim_path = create_simulation_directory(
        simulation_name=simulation_name,
        remote_parent_path=remote_path,
        local_parent_path=local_path,
        simulation_type="grey",
    )

    depth_map = depth_map[:, np.newaxis, :].astype(np.float64)
    domain = depth_map.shape

    max_depth = np.max(depth_map)

    porosity_map = depth_map / max_depth
    porosity_map = np.where(porosity_map == 1, 0.99999, porosity_map)
    porosity_map.astype(np.float64).tofile(os.path.join(local_sim_path, f"porosity_map_{simulation_name}.raw"))

    permeability_map = porosity_map * depth_map**2 / 12.0
    permeability_map.astype(np.float64).tofile(os.path.join(local_sim_path, f"permeability_map_{simulation_name}.raw"))

    depth_map = np.where(depth_map == 0, 0, 1)
    depth_map.astype(np.uint8).tofile(os.path.join(local_sim_path, f"{simulation_name}.raw"))

    common_db(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        resolution=resolution,
        component_labels=[0, 1],
        hardware_type=hardware_type,
        only_one_domain=only_one_domain,
    )

    greyscale_db(
        tau=tau,
        body_force=body_force,
        output_path=local_sim_path,
        simulation_name=simulation_name,
        timestepmax=timestepmax,
        tolerance=tolerance,
    )

    create_slurm_script(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        hardware_type=hardware_type,
        simulation_type="grey",
        only_one_domain=only_one_domain,
    )

    append_sbatch_command(
        local_parent_path=local_path,
        remote_simulation_path=remote_sim_path,
        simulation_name=simulation_name,
        simulation_type="grey",
    )


def fullscale_workspace(
    tau: float,
    body_force: float,
    depth_map: np.ndarray,
    local_path: str,
    remote_path: str,
    simulation_name: str,
    resolution: float,
    hardware_type: Literal["k40m", "a100", "cpu"],
    timestepmax: int,
    tolerance: float,
    only_one_domain: bool = False,
) -> None:
    """Orchestrates the workspace creation for fullscale LBPM permeability
        simulations.

    Args:
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        depth_map: A 2D NumPy array representing the dpeth distribution.
        local_path: The local parent directory to store the workspace.
        remote_path: The parent directory on the Linux cluster.
        simulation_name: Name of the simulation.
        resolution: Conversion factor in micrometers/voxel.
        hardware_type: The identifier for the hardware architecture .
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        None. The script creates the workspace directly in the file system.
    """  # noqa: D205
    local_sim_path, remote_sim_path = create_simulation_directory(
        simulation_name=simulation_name,
        remote_parent_path=remote_path,
        local_parent_path=local_path,
        simulation_type="full",
    )

    depth_voxels = np.round(depth_map / resolution).astype(int)
    max_voxels = np.max(depth_voxels)

    Ny = max_voxels + 2

    depth_voxels_3d = depth_voxels[:, np.newaxis, :]

    z_grid = np.arange(Ny)[np.newaxis, :, np.newaxis]

    full_image_bool = (z_grid > 0) & (z_grid <= depth_voxels_3d)

    full_image = full_image_bool.astype(np.uint8)

    domain = full_image.shape

    raw_file_path = os.path.join(local_sim_path, f"{simulation_name}.raw")
    full_image.astype(np.uint8).tofile(raw_file_path)

    common_db(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        resolution=resolution,
        component_labels=[0, 1],
        hardware_type=hardware_type,
        only_one_domain=only_one_domain,
    )

    fullscale_db(
        tau=tau,
        body_force=body_force,
        output_path=local_sim_path,
        simulation_name=simulation_name,
        timestepmax=timestepmax,
        tolerance=tolerance,
    )

    create_slurm_script(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        hardware_type=hardware_type,
        simulation_type="full",
        only_one_domain=only_one_domain,
    )

    append_sbatch_command(
        local_parent_path=local_path,
        remote_simulation_path=remote_sim_path,
        simulation_name=simulation_name,
        simulation_type="full",
    )


def laleian2015_workspace(
    tau: float,
    body_force: float,
    depth_map: np.ndarray,
    local_path: str,
    remote_path: str,
    simulation_name: str,
    resolution: float,
    timestepmax: int,
    tolerance: float,
) -> None:
    """Orchestrates the workspace creation for Python LBM simulation.

    Args:
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        depth_map: A 2D NumPy array representing the depth distribution.
        local_path: The local parent directory to store the workspace.
        remote_path: The parent directory on the Linux cluster.
        simulation_name: Name of the simulation.
        resolution: Conversion factor in micrometers/voxel.
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.

    Returns:
        None. The script creates the workspace directly in the file system.
    """  # noqa: D205
    local_sim_path, remote_sim_path = create_simulation_directory(
        simulation_name=simulation_name,
        remote_parent_path=remote_path,
        local_parent_path=local_path,
        simulation_type="laleian2015",
    )

    depth_map = np.round(depth_map / resolution) * resolution

    domain = depth_map.shape

    raw_file_path = os.path.join(local_sim_path, f"{simulation_name}.raw")
    depth_map.astype(float).tofile(raw_file_path)

    laleian2015_db(
        local_path=local_sim_path,
        remote_path=remote_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        resolution=resolution,
        tau=tau,
        body_force=body_force,
        timestepmax=timestepmax,
        tolerance=tolerance,
        on_cluster=True,
    )

    create_slurm_script(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        hardware_type="cpu",
        simulation_type="laleian2015",
        only_one_domain=True,
    )

    append_sbatch_command(
        local_parent_path=local_path,
        remote_simulation_path=remote_sim_path,
        simulation_name=simulation_name,
        simulation_type="laleian2015",
    )


def bidimensional_workspace(
    tau: float,
    body_force: float,
    depth_map: np.ndarray,
    local_path: str,
    remote_path: str,
    simulation_name: str,
    resolution: float,
    hardware_type: Literal["k40m", "a100", "cpu"],
    timestepmax: int,
    tolerance: float,
    only_one_domain: bool = False,
) -> None:
    """Orchestrates the workspace creation for 2D LBPM permeability
        simulations.

    Args:
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        depth_map: A 2D NumPy array representing the dpeth distribution.
        local_path: The local parent directory to store the workspace.
        remote_path: The parent directory on the Linux cluster.
        simulation_name: Name of the simulation.
        resolution: Conversion factor in micrometers/voxel.
        hardware_type: The identifier for the hardware architecture .
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        None. The script creates the workspace directly in the file system.
    """  # noqa: D205
    local_sim_path, remote_sim_path = create_simulation_directory(
        simulation_name=simulation_name,
        remote_parent_path=remote_path,
        local_parent_path=local_path,
        simulation_type="2D",
    )

    bidimensional_image = np.where(depth_map != 0, 1, 0).astype(int)

    bidimensional_3d = bidimensional_image[:, np.newaxis, :]

    domain = bidimensional_3d.shape

    raw_file_path = os.path.join(local_sim_path, f"{simulation_name}.raw")
    bidimensional_3d.astype(np.uint8).tofile(raw_file_path)

    common_db(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        resolution=resolution,
        component_labels=[0, 1],
        hardware_type=hardware_type,
        only_one_domain=only_one_domain,
    )

    fullscale_db(
        tau=tau,
        body_force=body_force,
        output_path=local_sim_path,
        simulation_name=simulation_name,
        timestepmax=timestepmax,
        tolerance=tolerance,
    )

    create_slurm_script(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        hardware_type=hardware_type,
        simulation_type="2D",
        only_one_domain=only_one_domain,
    )

    append_sbatch_command(
        local_parent_path=local_path,
        remote_simulation_path=remote_sim_path,
        simulation_name=simulation_name,
        simulation_type="2D",
    )


def format_identifier(value: float, symbol: str = "") -> str:
    """Formats a numeric value into a string identifier for filenames.

    Args:
        value: The numeric value to be formatted.
        symbol: An optional string symbol to insert after the underscore.

    Returns:
        The formatted string identifier.
    """
    if value.is_integer():
        return f"_{symbol}_{int(value)}"

    return f"_{symbol}_{str(value).replace('.', 'p')}"
