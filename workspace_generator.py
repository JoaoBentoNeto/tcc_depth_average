import os  # noqa: D100
import posixpath
import textwrap
from typing import Literal

import numpy as np


def create_simulation_directory(
    simulation_name: str,
    remote_parent_path: str,
    local_parent_path: str,
) -> tuple[str, str]:
    """Creates a local directory and generates local and remote paths.

    Args:
        simulation_name: The base name of the directory to be created.
        remote_parent_path: The parent path on the Linux cluster.
        local_parent_path: The local parent directory.


    Returns:
        A tuple containing two strings:
        - local_path: The absolute path to the created directory.
        - remote_path: The corresponding path formatted for Linux.
    """
    local_path = os.path.join(local_parent_path, simulation_name)
    remote_path = posixpath.join(remote_parent_path, simulation_name)

    os.makedirs(local_path, exist_ok=True)

    return local_path, remote_path


def divide_domain(
    domain: tuple[int, int, int],
    only_one_domain: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Finds the best geometric division to maximize gpu usage.

    Args:
        domain: A tuple containing the total volume dimensions (nx, ny, nz).
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

    for num_procs in [4, 3, 2, 1]:
        best_division = None
        low_score = float("inf")

        for n_procs_x in range(1, num_procs + 1):
            for n_procs_y in range(1, (num_procs // n_procs_x) + 1):
                n_procs_z = num_procs // (n_procs_x * n_procs_y)

                if n_procs_x * n_procs_y * n_procs_z == num_procs:
                    if (
                        nx % n_procs_x == 0
                        and ny % n_procs_y == 0
                        and nz % n_procs_z == 0
                    ):
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
    gpu_type: Literal["k40m", "a100", None],
    simulation_type: Literal["full", "grey", "autoral"],
    only_one_domain: bool = False,
) -> None:
    """Generates a SLURM batch script.

    Args:
        output_path: Directory path where the .sh script will be saved.
        simulation_name: Name of the simulation.
        domain: A tuple containing the total volume dimensions (nx, ny, nz).
        gpu_type: The identifier for the GPU architecture.
            Must be exactly "k40m", "a100" or None (python case).
        simulation_type: Defines the routine and environment to be used.
            Must be exactly "full", "grey", or "autoral".
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        None. The script is written directly to the file system.
    """
    script_path = os.path.join(output_path, f"{simulation_name}.sh")

    if simulation_type == "autoral":
        content = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={simulation_name}
            #SBATCH --output=output_%j.log
            #SBATCH --error=error_%j.log
            #SBATCH --partition=close_cpu
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=16
            #SBATCH --time=00:00:00

            module load conda
            conda activate depth_averaged
            python -u /home/joao.neto/TCC/lbm_run.py ./{simulation_name}.db
            """)
    else:
        n_procs, _ = divide_domain(domain, only_one_domain)
        total_processors = int(n_procs[0] * n_procs[1] * n_procs[2])

        if simulation_type == "full":
            routine = "lbpm_permeability_simulator"
            module = "lbpm/gpu"
        elif simulation_type == "grey":
            routine = "lbpm_greyscale_simulator"
            module = f"lbpm/gpu/poro_grey_fix_{gpu_type}"
        else:
            raise ValueError(f"Invalid simulation_type: {simulation_type}")

        content = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={simulation_name}
            #SBATCH --output=output_%j.log
            #SBATCH --error=error_%j.log
            #SBATCH --partition=all_gpu
            #SBATCH --nodes=1
            #SBATCH --ntasks={total_processors}
            #SBATCH --gres=gpu:{gpu_type}:{total_processors}
            #SBATCH --cpus-per-task=1
            #SBATCH --time=00:00:00

            module load {module}
            mpirun -np {total_processors} {routine} ./{simulation_name}.db
            """)

    with open(script_path, "w") as file:
        file.write(content)


def common_db(
    output_path: str,
    simulation_name: str,
    domain: tuple[int, int, int],
    resolution: float,
    component_labels: tuple[int, ...],
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
        only_one_domain: If True, restricts the subdomains division to a single
            domain.

    Returns:
        None. The script is written directly to the file system.
    """  # noqa: D205
    read_write_values = ", ".join(f"{label}" for label in component_labels)
    n_procs, subdomains = divide_domain(domain, only_one_domain)

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
            write_silo = true
            save_8bit_raw = true
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
    component_labels: tuple[int, ...],
    porosity_list: tuple[float, ...],
    permeability_list: tuple[float, ...],
) -> None:
    """Complements the database script for greyscale LBPM simulations.

    Args:
        tau: Relaxation time used in LBM simulations.
        body_force: Acceleration field that stimulates the fluid flow.
        output_path: Directory path where the .db script is located.
        simulation_name: Name of the simulation.
        timestepmax: The maximum number of timesteps for the simulation.
        tolerance: An absolute difference convergence parameter.
        component_labels: A tuple of integer labels representing the distinct
            phases in the domain (e.g., 0 for solid, others for porous media).
        porosity_list: A tuple with the porosity value corresponding to each
            component label.
        permeability_list: A tuple with the permeability value corresponding
            to each component label.

    Returns:
        None. The script appends the configuration directly to the file system.
    """
    str_labels = ", ".join(str(lbl) for lbl in component_labels if lbl != 0)
    str_porosities = ", ".join(
        f"{por:.8f}"
        for lbl, por in zip(component_labels, porosity_list)
        if lbl != 0
    )
    str_perms = ", ".join(
        f"{perm:.8f}"
        for lbl, perm in zip(component_labels, permeability_list)
        if lbl != 0
    )

    content = textwrap.dedent(f"""\
        
        Greyscale {{
            tau = {tau}
            F = 0.0, 0.0, {body_force}
            timestepMax = {timestepmax}
            tolerance = {tolerance}
            collision = "MRT"
            ComponentLabels = {str_labels}
            PorosityList = {str_porosities}
            PermeabilityList = {str_perms}
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


def autoral_db(
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
        LBM Depth-averaged Autoral {{
            tau = {tau}
            F = {body_force}, 0.0
            timestepMax = {timestepmax}
            tolerance = {tolerance}
            local = "{simulation_path}"
            nome = "{simulation_name}"
            resolucao = {resolution}
            dimensions = {domain[1]}, {domain[0]}
        }}""")

    with open(os.path.join(local_path, f"{simulation_name}.db"), "w") as file:
        file.write(content)


def append_sbatch_command(
    local_parent_path: str,
    remote_simulation_path: str,
    simulation_name: str,
    simulation_type: Literal["full", "grey", "autoral"],
) -> None:
    """Appends the submission command to a type-specific sbatches text file.

    If the master file (e.g., 'sbatches_greyscale.txt') does not exist in
    the local parent path, it will be automatically created.

    Args:
        local_parent_path: The local directory where the .txt will be saved.
        remote_simulation_path: The path of the simulation on the cluster.
        simulation_name: The name of the simulation job.
        simulation_type: Defines the target master file. Must be exactly
            "full", "grey", or "autoral".

    Raises:
        ValueError: If an invalid simulation_type is provided.

    Returns:
        None. The command is written directly to the file.
    """
    if simulation_type == "full":
        txt_filename = "sbatches_fullscale.txt"
    elif simulation_type == "grey":
        txt_filename = "sbatches_greyscale.txt"
    elif simulation_type == "autoral":
        txt_filename = "sbatches_autoral.txt"
    else:
        raise ValueError(f"Invalid simulation_type: {simulation_type}")

    txt_path = os.path.join(local_parent_path, txt_filename)

    command = (
        f"cd {remote_simulation_path} && "
        f"dos2unix {simulation_name}.sh && "
        f"sbatch {simulation_name}.sh\n"
    )

    with open(txt_path, "a") as file:
        file.write(command)


def greyscale_workspace(
    tau: float,
    body_force: float,
    depth_map: np.ndarray,
    local_path: str,
    remote_path: str,
    simulation_name: str,
    resolution: float,
    gpu_type: Literal["k40m", "a100"],
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
        gpu_type: The identifier for the GPU architecture .
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
    )

    depth_map = np.round(depth_map / resolution) * resolution
    max_depth = np.round(np.max(depth_map) / resolution) * resolution

    values = np.unique(depth_map)
    values = values[values != 0.0]

    label_map = np.zeros_like(depth_map, dtype=int)

    n_groups = min(len(values), 126)

    groups = np.array_split(values, n_groups)

    depth_array = np.array([np.mean(g) for g in groups])
    depth_array = np.concatenate(([0.0], depth_array))
    depth_list = depth_array.tolist()

    if n_groups > 1:
        bins = [g[0] for g in groups[1:]]
        mask = depth_map != 0.0

        label_map[mask] = np.digitize(depth_map[mask], bins) + 1
    else:
        label_map[depth_map != 0.0] = 1

    component_labels = list(range(len(depth_list)))

    porosity_list = np.where(
        depth_array / max_depth >= 1, 0.99999, depth_array / max_depth
    )
    permeability_list = porosity_list * (depth_array**2) / 12

    image_3d = label_map[:, np.newaxis, :]
    domain = image_3d.shape

    raw_file_path = os.path.join(local_sim_path, f"{simulation_name}.raw")
    image_3d.astype(np.uint8).tofile(raw_file_path)

    common_db(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        resolution=resolution,
        component_labels=component_labels,
        only_one_domain=only_one_domain,
    )

    greyscale_db(
        tau=tau,
        body_force=body_force,
        output_path=local_sim_path,
        simulation_name=simulation_name,
        timestepmax=timestepmax,
        tolerance=tolerance,
        component_labels=component_labels,
        porosity_list=porosity_list,
        permeability_list=permeability_list,
    )

    create_slurm_script(
        output_path=local_sim_path,
        simulation_name=simulation_name,
        domain=domain,
        gpu_type=gpu_type,
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
    gpu_type: Literal["k40m", "a100"],
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
        gpu_type: The identifier for the GPU architecture .
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
        gpu_type=gpu_type,
        simulation_type="full",
        only_one_domain=only_one_domain,
    )

    append_sbatch_command(
        local_parent_path=local_path,
        remote_simulation_path=remote_sim_path,
        simulation_name=simulation_name,
        simulation_type="full",
    )


def autoral_workspace(
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
    )

    depth_map = np.round(depth_map / resolution) * resolution

    domain = depth_map.shape

    raw_file_path = os.path.join(local_sim_path, f"{simulation_name}.raw")
    depth_map.astype(np.uint8).tofile(raw_file_path)

    autoral_db(
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
        gpu_type=None,
        simulation_type="autoral",
        only_one_domain=True,
    )

    append_sbatch_command(
        local_parent_path=local_path,
        remote_simulation_path=remote_sim_path,
        simulation_name=simulation_name,
        simulation_type="autoral",
    )

# 1. Adiciona as mudanças recentes na área de preparo
git add workspace_generator.py

# 2. Faz o commit com título e corpo detalhado
git commit -m "feat: finalized workspace generator" -m "Implement functions for orchestrating greyscale, fullscale, and autoral depth-averaged LBM simulation workspaces. Standardize parameter naming conventions across all modules."

# 3. Envia o código para o repositório no GitHub
git push