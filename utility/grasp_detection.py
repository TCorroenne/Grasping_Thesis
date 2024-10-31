from skimage.feature import peak_local_max
from utility.grasp import Grasp

def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1, mask=None,threshold=0.2):
    """
    Detect grasps in a GG-CNN output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=threshold, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        if mask is not None and not mask[grasp_point]:
            # Invalid grasp point, not in mask
            continue

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length/2

        grasps.append(g)

    return grasps

def detect_grasp_with_quality(q_img, ang_img, width_img, no_grasps=1, mask=None,threshold=0.2):
    """
    Detect grasps in a GG-CNN output, with a quality measure.
    Args:
        q_img: np.array: Q image network output.
        ang_img: np.array: Angle image network output.
        width_img: np.array: Width image network output.
        no_grasps: int: Max number of grasps to return.
        mask: np.array: Mask for valid grasps.
        threshold: float: Threshold for grasps.
    Returns:
        list: List of (grasps,quality) tuples. 
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=threshold, num_peaks=no_grasps)
    grasps_q = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        if mask is not None and not mask[grasp_point]:
            # Invalid grasp point, not in mask
            continue

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length/2

        grasps_q.append((g, q_img[grasp_point]))
    # Sort by quality (high to low)
    grasps_q = sorted(grasps_q, key=lambda x: x[1], reverse=True)
    return grasps_q