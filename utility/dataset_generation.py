import cv2
import numpy as np
import sys
import glob
import re
import matplotlib.pyplot as plt
import shutil
sys.path.append(".") # Adds higher directory to python modules path.
from utility.data_processing import *
from utility.saving import save_results_cornell
# from PEGG_Net.utils.dataset_processing.image import DepthImage

def get_sparse_pure_dataset(imput_path, output_path):
    graspf = glob.glob(os.path.join(imput_path, '*', 'pcd*cpos_s[5,6,7,8,9]*.txt'))
    print('Found {} positive samples'.format(len(graspf)))
    depthf = []
    rgbf = []
    scoresf = []
    for f in graspf:
        match = re.search(r"cpos_s", f)
        start_match = match.start()
        end_match = match.end()
        depthf.append(f[:start_match] + "d.tiff")
        rgbf.append(f[:start_match] + "r.png")
        scoresf.append(int(f[end_match]))
    out_files = output_path + "/01/"
    if not os.path.exists(out_files):
        os.makedirs(out_files)
    for i in range(len(graspf)):
        str_i = str(i)
        if len(str_i) == 1:
            str_i = "0"+str_i
        shutil.copyfile(graspf[i], out_files + "pcd" + str_i + "cpos_s" + str(scoresf[i]) + ".txt")
        shutil.copyfile(depthf[i], out_files + "pcd" + str_i + "d.tiff")
        shutil.copyfile(rgbf[i], out_files + "pcd" + str_i + "r.png")

def get_annotated_dataset(input_path, output_path):
    graspf = glob.glob(os.path.join(input_path, '*', 'pcd*cpos_s*.txt'))
    print('Found {} samples'.format(len(graspf)))
    depthf = []
    rgbf = []
    scoresf = []
    for f in graspf:
        match = re.search(r"cpos_s", f)
        start_match = match.start()
        end_match = match.end()
        depthf.append(f[:start_match] + "d.tiff")
        rgbf.append(f[:start_match] + "r.png")
        scoresf.append(int(f[end_match]))
    out_files = output_path + "/01/"
    if not os.path.exists(out_files):
        os.makedirs(out_files)
    for i in range(len(graspf)):
        str_i = str(i)
        if len(str_i) == 1:
            str_i = "0"+str_i
        depth_img = cv2.imread(depthf[i], cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(rgbf[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        grasps = get_n_grasps_on_images(image, n=5)
        if scoresf[i] >=7:
            shutil.copyfile(graspf[i], out_files + "pcd" + "01" + str_i + "cpos_s" + str(scoresf[i]) + ".txt")
            save_results_cornell(grasps, rgb_img, depth_img, output_path, grasp_points_suffix="cpos_s"+str(scoresf[i]), subfolder="01", index_str=str_i)
        else:
            save_results_cornell(grasps, rgb_img, depth_img, output_path, grasp_points_suffix="cpos_s"+"5", subfolder="01", index_str=str_i)
    
# get_sparse_pure_dataset('dataset_generation/validation_base', 'dataset_generation/validation_sparse')
# get_annotated_dataset('dataset_generation/validation_base', 'dataset_generation/validation_annotated')