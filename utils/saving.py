import matplotlib.pyplot as plt
import numpy as np
from utils.grasp_detection import detect_grasps
from utils.grasp import Grasp, GraspRectangle
import cv2

def save_results(rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None, save_dir='results', name = 'grasp'):
    """
    Save the results of a grasp detection.
    Args:
        rgb_img: np.array: The RGB image.
        grasp_q_img: np.array: The grasp quality image.
        grasp_angle_img: np.array: The grasp angle image.
        depth_img: np.array: The depth image.
        no_grasps: int: The number of grasps to plot.
        grasp_width_img: np.array: The grasp width image.
        save_dir: str: The directory to save the results in.
        name: str: The name of the results.
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    
    save_prefix = save_dir + '/' + name + '_'
    
    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')
    fig.savefig(save_prefix + 'rgb.png')

    if depth_img.any():
        fig = plt.figure(figsize=(10, 10))
        plt.ion()
        plt.clf()
        ax = plt.subplot(111)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')
        fig.savefig(save_prefix + 'depth.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')
    fig.savefig(save_prefix + 'grasp.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig(save_prefix + 'quality.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig(save_prefix + 'angle.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig(save_prefix + 'width.png')

    fig.canvas.draw()
    plt.close(fig)
    
def save_results_cornell(attemptedGrasp, rgb_img, depth_img, save_dir='results_cornell', grasp_points_suffix='cpos',subfolder='01', index_str='00'):
    """
    Save the results of a grasp detection in the Cornell dataset format.
    Args:
        attemptedGrasp: Grasp or GraspRectangle: The attempted grasp.
        rgb_img: np.array: The RGB image.
        depth_img: np.array: The depth image.
        save_dir: str: The directory to save the results in.
        grasp_points_suffix: str: The suffix for the grasp points should be cpos or cneg
    """
    if isinstance(attemptedGrasp, Grasp):
        grasp_rectangle = attemptedGrasp.as_gr
    else:
        grasp_rectangle = attemptedGrasp
    base_save_dir = save_dir + '/' + subfolder + '/' + 'pcd' + subfolder + index_str
    grasp_points_file = open(base_save_dir + grasp_points_suffix + '.txt', 'a')
    for point in grasp_rectangle.points:
        grasp_points_file.write(str(point[1]) + ' ' + str(point[0]) + '\n')
    grasp_points_file.close()
    # Save the depth image as a tiff
    cv2.imwrite(base_save_dir + 'd.tiff', depth_img)
    # Save the rgb image as png
    cv2.imwrite(base_save_dir + 'r.png', rgb_img)
    