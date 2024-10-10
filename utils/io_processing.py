import cv2
import numpy as np
from skimage.filters import gaussian
import torch


def rgbd_input_processing(rgb_img,depth_img,use_depth=True,use_rgb=True):
    """
    Preprocesses the input images for the model.
    Args:
        rgb_img: np.array: The RGB image, assumed to be the right size.
        depth_img: np.array: The depth image, assumed to be the right size.
    Returns:
        np.array: The torch input to send to the model.
    """
    # process the images
    rgb_img = process_rgb(rgb_img)
    depth_img = process_depth(depth_img)
    out_size = depth_img.shape[0]
    # Stack
    if use_rgb and not use_depth:
        input_img = rgb_img.reshape(1, 1, out_size, out_size)
    elif use_depth and not use_rgb:
        input_img = depth_img.reshape(1, 1, out_size, out_size)
    else:
        print(rgb_img.shape)
        print(depth_img.shape)
        input_img = np.concatenate((np.expand_dims(depth_img, 0), rgb_img), 0).reshape(1, 4, out_size, out_size)
        print(input_img.shape)
    return numpy_to_torch(input_img)

def get_0_1_float32(img):
    """
    Converts an image to float32, with values between 0 and 1.
    Args:
        img: np.array: The image to convert.
    Returns:
        np.array: The converted image.
        int: The maximum value in the image.
    """
    max_val = img.max()
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    else:
        img = img.astype(np.float32) / max_val
    return img, max_val

def process_depth(depth_img):
    """
    Process the depth image.
    Args:
        depth_img: np.array: The depth image.
    Returns:
        np.array: The processed depth image.
    """
    # Convert to float
    if depth_img.dtype == np.uint8:
        depth_img, max_depth = get_0_1_float32(depth_img)
    # Center on mean
    depth_img = depth_img - depth_img.mean()
    # Clamp to [-1, 1]
    depth_img = np.clip(depth_img, -1, 1)
    return depth_img

def process_rgb(rgb_img):
    """
    Process the RGB image.
    Args:
        rgb_img: np.array: The RGB image.
    Returns:
        np.array: The processed RGB image.
    """
    # Convert to float
    rgb_img, max_rgb = get_0_1_float32(rgb_img)
    # Center on mean
    rgb_img = rgb_img - rgb_img.mean()
    # Clamp to [-1, 1] (should be already the case, even between 0 and 1)
    rgb_img = np.clip(rgb_img, -1, 1)
    return rgb_img.transpose((2, 0, 1))

def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
def process_raw_output(pos_output, cos_output, sin_output, width_output, depth_nan=None, width_factor=150.0):
    """
    Process the raw output of the model.
    Args:
        output: np.array: The raw output of the model.
    Returns:
        np.array: The processed output.
    """
    q_img = pos_output.cpu().numpy().squeeze()
    if depth_nan is not None:
        q_img[depth_nan] = 0
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = (torch.atan2(sin_output, cos_output) / 2.0).cpu().numpy().squeeze()
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = width_output.cpu().numpy().squeeze() * width_factor
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    return q_img, ang_img, width_img