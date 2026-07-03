import numpy as np
import porespy as ps
import skfmm
from scipy.ndimage import distance_transform_edt, label, maximum_filter
from skimage.morphology import medial_axis as ski_medial_axis
from skimage.morphology import skeletonize as ski_skeletonize


def calculate_skeleton_map(depth_map: np.ndarray) -> np.ndarray:
    """Calculates the semi-width map of a fluid domain.

    Applies a combination of distance transform and skeletonization with
    a maximum filter to accurately map the semi-width of complex fluid channels.
    Uses dynamic edge padding to repeat boundary pixels, avoiding boundary
    artifacts and minimizing memory usage. Includes debugging plots.

    Args:
        depth_map: A numpy array (2D or batched N-D) representing the domain,
            where 0 indicates solid and non-zero values indicate fluid.

    Returns:
        A numpy array of the same shape as `depth_map` containing the calculated
        semi-width values for the fluid pixels, and 0 for the solid pixels.
    """
    pore_radius = ps.filters.local_thickness(depth_map != 0)
    pad_size = int(np.round(np.max(pore_radius) * 2))
    pad_axes = (0, 1)
    pad_width = []

    for i in range(depth_map.ndim):
        if i in pad_axes:
            first_slice = np.take(depth_map, 0, axis=i)
            pad_before = 0 if np.all(first_slice == 0) else pad_size

            last_slice = np.take(depth_map, -1, axis=i)
            pad_after = 0 if np.all(last_slice == 0) else pad_size

            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))

    padded_depth_map = np.pad(depth_map, pad_width, mode="wrap")

    fluid = padded_depth_map != 0
    del padded_depth_map

    distance_map = distance_transform_edt(fluid)
    skeleton = ski_skeletonize(fluid)

    max_dist = maximum_filter(distance_map, size=3)
    distance_map[skeleton] = max_dist[skeleton]
    del max_dist

    indices = distance_transform_edt(skeleton == 0, return_indices=True)[1]
    del skeleton

    spread_map = distance_map[tuple(indices)]
    del indices
    del distance_map

    slices = []
    for before, after in pad_width:
        start = before if before > 0 else None
        end = -after if after > 0 else None
        slices.append(slice(start, end))

    final_map = spread_map[tuple(slices)].copy()

    original_fluid = depth_map != 0
    final_map *= original_fluid

    final = np.where(depth_map != 0, final_map, np.nan)
    return final


def calculate_local_thickness_map(depth_map: np.ndarray) -> np.ndarray:
    """Calculates the semi-width map of a fluid domain using Local Thickness.

    Applies the PoreSpy local thickness filter (maximal inscribed spheres).
    By its geometric nature, this method natively handles open boundaries
    (for channels longer than their width) without the need for artificial
    padding.

    Args:
        depth_map: A numpy array (2D or batched N-D) representing the domain,
            where 0 indicates solid and non-zero values indicate fluid.

    Returns:
        A numpy array of the same shape as `depth_map` containing the calculated
        local thickness values for the fluid pixels, and np.nan for the solid pixels.
    """
    fluid = depth_map != 0

    thickness_map = ps.filters.local_thickness(fluid)

    final_map = np.where(fluid, thickness_map, np.nan)

    final = np.where(depth_map != 0, final_map, np.nan)

    return final


def calculate_medial_axis_map(depth_map: np.ndarray) -> np.ndarray:
    """Calculates the semi-width map of a fluid domain using Medial Axis.

    Applies the medial axis transform to accurately map the semi-width.
    Uses dynamic symmetric padding determined by the minimum image dimension
    to avoid boundary artifacts and exploding memory in purely solid boundaries.

    Args:
        depth_map: A numpy array (2D or batched N-D) representing the domain,
            where 0 indicates solid and non-zero values indicate fluid.

    Returns:
        A numpy array of the same shape as `depth_map` containing the calculated
        semi-width values for the fluid pixels, and np.nan for the solid pixels.
    """

    pore_radius = ps.filters.local_thickness(depth_map != 0)
    pad_size = int(np.round(np.max(pore_radius) * 2))
    pad_axes = (0, 1)
    pad_width = []

    for i in range(depth_map.ndim):
        if i in pad_axes:
            first_slice = np.take(depth_map, 0, axis=i)
            pad_before = 0 if np.all(first_slice == 0) else pad_size

            last_slice = np.take(depth_map, -1, axis=i)
            pad_after = 0 if np.all(last_slice == 0) else pad_size

            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))

    padded_depth_map = np.pad(depth_map, pad_width, mode="wrap")

    fluid = padded_depth_map != 0
    del padded_depth_map

    esqueleto, d_esqueleto = ski_medial_axis(fluid, return_distance=True)

    indices = distance_transform_edt(esqueleto == 0, return_indices=True)[1]
    del esqueleto

    spread_map = d_esqueleto[tuple(indices)]
    del indices
    del d_esqueleto

    slices = []
    for before, after in pad_width:
        start = before if before > 0 else None
        end = -after if after > 0 else None
        slices.append(slice(start, end))

    cropped_map = spread_map[tuple(slices)].copy()
    del spread_map

    original_fluid = depth_map != 0
    final_map = np.where(original_fluid, cropped_map, np.nan)

    final = np.where(depth_map != 0, final_map, np.nan)

    return final


def calculate_n_edt_map(depth_map: np.ndarray) -> np.ndarray:
    """Calculates the semi-width map of a fluid domain using Geodesic N-EDT.

    Applies the Fast Marching Method (FMM) to calculate physical distances around
    solid obstacles. Uses dynamic symmetric padding determined by the minimum
    image dimension to avoid boundary artifacts. All physics are calculated on
    the padded domain, and cropped only at the final step.

    Args:
        depth_map: A numpy array (2D or batched N-D) representing the domain,
            where 0 indicates solid and non-zero values indicate fluid.

    Returns:
        A numpy array of the same shape as `depth_map` containing the calculated
        semi-width values for the fluid pixels, and np.nan for the solid pixels.
    """

    pore_radius = ps.filters.local_thickness(depth_map != 0)
    pad_size = int(np.round(np.max(pore_radius) * 2))
    pad_axes = (0, 1)
    pad_width = []

    for i in range(depth_map.ndim):
        if i in pad_axes:
            first_slice = np.take(depth_map, 0, axis=i)
            pad_before = 0 if np.all(first_slice == 0) else pad_size

            last_slice = np.take(depth_map, -1, axis=i)
            pad_after = 0 if np.all(last_slice == 0) else pad_size

            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))

    padded_depth_map = np.pad(depth_map, pad_width, mode="wrap")

    fluid_padded = padded_depth_map != 0
    solid_padded = ~fluid_padded
    del padded_depth_map

    labeled_solid, num_solids = label(solid_padded)

    all_edts = np.zeros((num_solids,) + fluid_padded.shape, dtype=np.float64)

    for i in range(1, num_solids + 1):
        phi = np.ones(fluid_padded.shape, dtype=np.float64)
        phi[labeled_solid == i] = 0.0

        barreiras = (labeled_solid != i) & (labeled_solid > 0)
        phi_masked = np.ma.MaskedArray(phi, mask=barreiras)

        try:
            dist_geodesica = skfmm.distance(phi_masked)
            all_edts[i - 1] = dist_geodesica.filled(np.inf)
        except ValueError:
            all_edts[i - 1] = np.inf

    if num_solids >= 2:
        menores = np.partition(all_edts, 1, axis=0)[:2]
        dist_1 = menores[0]
        dist_2 = menores[1]

        fallback_dist = distance_transform_edt(fluid_padded) - 0.5

        w_half_map_padded = np.where(
            dist_2 != np.inf, ((dist_1 + dist_2) / 2.0) - 0.5, fallback_dist
        )
    else:
        w_half_map_padded = distance_transform_edt(fluid_padded) - 0.5

    slices = []
    for before, after in pad_width:
        start = before if before > 0 else None
        end = -after if after > 0 else None
        slices.append(slice(start, end))

    cropped_map = w_half_map_padded[tuple(slices)].copy()
    del w_half_map_padded

    original_fluid = depth_map != 0
    final_map = np.where(original_fluid, np.maximum(cropped_map, 0.0), np.nan)

    final = np.where(depth_map != 0, final_map, np.nan)

    return final


def calculate_shear_permeability(
    semi_width_map: np.ndarray,
    depth_map: np.ndarray | float,
    resolution: float,
) -> np.ndarray:
    """Calculates the optimized shear permeability map for a fluid domain.

    Uses a memory-efficient loop over odd wave numbers to compute
    the local permeability based on the channel's semi-width and depth,
    computing only the active fluid nodes.

    Args:
        semi_width_map: A 2D numpy array representing the local semi-width
            of the channel.
        depth_map: A 2D numpy array or float representing the local depth
            of the channel.

    Returns:
        A numpy array containing the computed permeability values for the fluid
        regions, and 1.0 for the solid regions.
    """
    fluid_mask = depth_map != 0

    distance_map = distance_transform_edt(fluid_mask)

    edt = distance_map[fluid_mask] * resolution
    semi_width = semi_width_map[fluid_mask] * resolution

    if isinstance(depth_map, np.ndarray):
        depth = depth_map[fluid_mask]
    else:
        depth = depth_map

    del distance_map

    semi_width = np.maximum(edt, semi_width)

    num_sum = np.zeros_like(edt, dtype=np.float64)
    den_sum = np.zeros_like(edt, dtype=np.float64)

    for n_val in range(1, 2500, 2):
        n = float(n_val)

        pi_n_over_d = (np.pi * n) / depth

        arg1 = -pi_n_over_d * edt
        arg2 = -pi_n_over_d * (2.0 * semi_width - edt)
        arg3 = -pi_n_over_d * 2.0 * semi_width

        ratio = (np.exp(arg1) + np.exp(arg2)) / (1.0 + np.exp(arg3))

        n2 = n * n
        n4 = n2 * n2
        den_sum += ratio / n2
        num_sum += ratio / n4

    numerator = (np.pi**4 / 96.0) - num_sum
    denominator = (np.pi**2 / 8.0) - den_sum

    perm_fluid = (depth**2 / np.pi**2) * (numerator / denominator)

    permeability = np.ones_like(semi_width_map, dtype=np.float64)

    permeability[fluid_mask] = perm_fluid

    return permeability
