import glob
import os
from utility.grasp import GraspRectangles
from utility.grasp_detection import detect_grasps


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False
    
def process_raw_dataset(input_dataset_path, output_dataset_path):
    """
    Process a raw dataset into a format suitable for training the GG-CNN.
    """
    graspf_10 = glob.glob(os.path.join(input_dataset_path, '*', 'pcd*cpos_s10.txt'))
    grarspf_5_9 = glob.glob(os.path.join(input_dataset_path, '*', 'pcd*cpos_s0[5-9].txt'))
    graspf_pos = graspf_10 + grarspf_5_9
    graspf_bad = glob.glob(os.path.join(input_dataset_path, '*', 'pcd*bad.txt'))
    graspf_0_4 = glob.glob(os.path.join(input_dataset_path, '*', 'pcd*cpos_s0[0-4].txt'))
    graspf_neg = graspf_bad + graspf_0_4
    print('Found {} positive and {} negative samples'.format(len(graspf_pos), len(graspf_neg)))
    print(graspf_pos)
    print(graspf_neg)
