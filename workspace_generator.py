import os
import textwrap
from typing import Optional


def create_simulation_directory(
    base_name: str, parent_path: Optional[str] = None, overwrite: bool = True
) -> str:
    """Creates a directory to store simulation files.

    Args:
        base_name: The base name of the directory to be created.
        parent_path: The parent directory path. If None, uses the
            current working directory.
        overwrite: If False, appends a numeric suffix to the directory
            name if it already exists, preventing data loss.

    Returns:
        directory: The absolute path to the created directory.
    """
    if parent_path is None:
        directory = base_name
    else:
        directory = os.path.join(parent_path, base_name)

    if not overwrite:
        counter = 0
        while os.path.exists(directory):
            counter += 1
            if parent_path is None:
                directory = f"{base_name}_{counter}"
            else:
                directory = os.path.join(parent_path, f"{base_name}_{counter}")

    os.makedirs(directory, exist_ok=True)

    return directory


def divide_domain(
    domain: tuple[int, int, int],
    only_one_domain: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Finds the best geometric division to maximize gpu usage.

    Args:
        domain: A tuple containing the total volume dimensions (nx, ny, nz).
        only_one_domain: If True, bypasses the division algorithm and
            returns the full geometry for a single processor.

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


from typing import Literal, Optional


def create_slurm_script(
    output_dir: str,
    job_name: str,
    domain: tuple[int, int, int],
    gpu_type: Literal["k40m", "a100"],
    simulation_type: Literal["full", "grey", "autoral"],
    only_one_domain: bool = False,
) -> None:
    """Generates a SLURM batch script for LBM simulations.

    Args:
        output_dir: Directory path where the .sh script will be saved.
        job_name: Name of the job, used for the script file and SLURM.
        domain: A tuple containing the total volume dimensions (nx, ny, nz).
        gpu_type: The identifier for the GPU architecture.
            Must be exactly "k40m" or "a100".
        simulation_type: Defines the routine and environment to be used.
            Must be exactly "full", "grey", or "autoral".
        only_one_domain: If True, restricts the subdomains division to a single domain.

    Returns:
        None. The script is written directly to the file system.
    """
    script_path = os.path.join(output_dir, f"{job_name}.sh")

    if simulation_type == "autoral":
        content = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH --output=output_%j.log
            #SBATCH --error=error_%j.log
            #SBATCH --partition=close_cpu
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=16
            #SBATCH --time=00:00:00

            module load conda
            conda activate depth_averaged
            python -u /home/joao.neto/depth_average_clean/lbm_run.py ./{job_name}.db
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
            #SBATCH --job-name={job_name}
            #SBATCH --output=output_%j.log
            #SBATCH --error=error_%j.log
            #SBATCH --partition=all_gpu
            #SBATCH --nodes=1
            #SBATCH --ntasks={total_processors}
            #SBATCH --gres=gpu:{gpu_type}:{total_processors}
            #SBATCH --cpus-per-task=1
            #SBATCH --time=00:00:00

            module load {module}
            mpirun -np {total_processors} {routine} ./{job_name}.db
            """)

    with open(script_path, "w") as file:
        file.write(content)

git commit -m "Initial commit: firsts functions to setup the workspace for the simulations later"