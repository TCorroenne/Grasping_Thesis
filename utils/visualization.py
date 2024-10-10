import matplotlib.pyplot as plt
from utils.grasp_detection import detect_grasps
import numpy as np

def plot_output_full(rgb_img, depth_img, grasp_q_img, grasp_angle_img,grasp_width_img, no_grasps=1, ):
    """
    Plot the output of a GG-CNN
    Args:
        rgb_img: np.array: The RGB image.
        depth_img: np.array: The depth image.
        grasp_q_img: np.array: The grasp quality image.
        grasp_angle_img: np.array: The grasp angle image.
        no_grasps: int: The number of grasps to plot.
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    
    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=155)
    ax.set_title('Width')
    ax.axis('off')
    
    plt.colorbar(plot)
    plt.show()