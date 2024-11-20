import glob
import os
import matplotlib.pyplot as plt
import utility.grasp as grasp
import numpy as np
import time 

def get_success_rate(dataset_path):
    """
    Get the success rate of the test trials
    """
    # graspf = glob.glob(os.path.join(dataset_path, '*', 'pcd*cpos.txt'))
    graspf_pos = glob.glob(os.path.join(dataset_path, '*', 'pcd*cpos_s[5-9].txt'))
    graspf_neg = glob.glob(os.path.join(dataset_path, '*', 'pcd*cpos_s[1-4].txt'))
    print('Found {} positive and {} negative samples'.format(len(graspf_pos), len(graspf_neg)))
    return len(graspf_pos)/(len(graspf_pos)+len(graspf_neg))

def get_number_score(dataset_path):
    """
    get the number of samples for each score
    """
    graspf_all = glob.glob(os.path.join(dataset_path, '*', 'pcd*cpos_s?.txt'))
    list_score = [0]*10
    for graspf in graspf_all:
        score = int(graspf[-5])
        list_score[score] += 1
    print(list_score)
    return list_score

def get_2_clicked_points_on_image(image):
    """ 
    Get 2 clicked points on an image using (x,y) coordinates
    """
    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    p1 = np.zeros((2,))
    p2 = np.zeros((2,))
    
    # Function to capture click event
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:  # Check if click is within the image
            # print(f"Clicked at: ({event.xdata} : {round(event.xdata)}, {event.ydata} : {round(event.ydata)})")
            if p1.all() == 0:
                p1[0] = round(event.xdata)
                p1[1] = round(event.ydata)
            else:
                p2[0] = round(event.xdata)
                p2[1] = round(event.ydata)
                # print(f"Points: {p1}, {p2}")
                plt.close()
    # Connect the click event to the function
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return p1.astype(int), p2.astype(int)

def get_grasp_from_2_points(p1, p2):
    """
    Get a grasp from 2 points in the (x,y) coordinate system
    """
    # We need to change p1 and p2 to (y,x) coordinate system
    p1 = np.flip(p1)
    p2 = np.flip(p2)
    center = np.mean([p1, p2], axis=0)
    length = np.linalg.norm(p1 - p2)
    angle = np.arctan2(p1[0] - p2[0], p2[1] - p1[1])
    width = length/2
    return grasp.Grasp(center, angle, length, width)

def get_n_grasps_on_images(image, n=10):
    """
    Get n grasps on an image
    """
    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    grasps = []
    global p1, p2
    p1 = np.zeros((2,))
    p2 = np.zeros((2,))
    
    # Function to capture click event
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:  # Check if click is within the image
            # print(f"Clicked at: ({event.xdata} : {round(event.xdata)}, {event.ydata} : {round(event.ydata)})")
            if len(grasps) == n:
                plt.close()
            if p1.all() == 0:
                p1[0] = round(event.xdata)
                p1[1] = round(event.ydata)
            else:
                p2[0] = round(event.xdata)
                p2[1] = round(event.ydata)
                # print(f"Points: {p1}, {p2}")
                gr = get_grasp_from_2_points(p1, p2)
                gr.plot(ax)
                grasps.append(gr)
                plt.show()
                p1[0] = 0
                p1[1] = 0
            
    # Connect the click event to the function
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    print("Grasps: ", grasps)
    print("last p1, p2: ", p1, p2) 
    return grasps
                
            