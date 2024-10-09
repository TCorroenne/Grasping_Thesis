import cv2
import numpy as np
from skimage.filters import gaussian

def extract_mask_from_img(img, threshold = 127, lower_than_threshold = True):
    """
    Extracts a mask from an image based on a threshold, in grayscale.
    Args:
        img: np.array: The image to extract the mask from.
        threshold: int: The threshold to use.
        lower_than_threshold: bool: If True, the mask will be True where the image is lower than the threshold.
    """
    if img.ndim == 3:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = img
    if lower_than_threshold:
        _, mask = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        _, mask = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY)
    return mask

def extend_mask(mask, extend_by=51, shape_square=False):
    """
    Extends a mask by a certain number of pixels.
    Args:
        mask: np.array: The mask to extend.
        extend_by: int: The number of pixels to extend the mask by.
        shape_square: bool: If True, the mask will be extended in a square shape.
    """
    shape_cv2 = cv2.MORPH_RECT if shape_square else cv2.MORPH_ELLIPSE
    # make sure extend_by is odd
    if extend_by%2 == 0:
        extend_by += 1
    element = cv2.getStructuringElement(shape_cv2, (extend_by, extend_by))
    mask_extended = cv2.dilate(mask, element)
    return mask_extended

def get_center_of_mass(mask):
    """
    Get the center of mass of a mask.
    Args:
        mask: np.array: The mask to get the center of mass of.
    Returns:
        int, int: The x (column) and y (line) coordinates of the center of mass.
    """
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def crop(img, cX, cY, crop_size):
    """
    Crop an image around a center point.
    Args:
        img: np.array: The image to crop.
        cX: int: The x coordinate of the center, or the column in matrix notation. 
        cY: int: The y coordinate of the center, or the line in in matrix notation.
        crop_size: int: The size of the crop.
    Returns:
        np.array: The cropped image.
    """
    shape = img.shape
    left = max(0, min(cX - crop_size//2, shape[1] - crop_size))
    top = max(0, min(cY - crop_size//2, shape[0] - crop_size))
    bottom = min(shape[0], top + crop_size)
    right = min(shape[1], left + crop_size)
    return img[top:bottom, left:right]

def process_rgbd(rgb_img, depth_img, mask=None, output_size=480, threshold_object_table=None):
    """
    Process an RGBD image.
    Args:
        rgb_img: np.array: The RGB image.
        depth_img: np.array: The depth image.
        mask: np.array: The mask to apply to the depth image.
        output_size: int: The size of the output image.
        threshold_object_table: int: The threshold to use to separate the object from the table and reconstruct a 'perfect' table.
    """
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    depth_nan = np.isnan(depth_img)
    depth_img[depth_nan] = 0
    if mask is None:
        #! THRESHOLD VALUES MIGHT NEED TO BE CHANGED
        mask = extract_mask_from_img(rgb_img, threshold=1, lower_than_threshold=False)
        mask = extend_mask(mask, extend_by=101, shape_square=False)
    depth_img = cv2.bitwise_or(depth_img, depth_img, mask=mask)

    cX, cY = get_center_of_mass(mask)
    depth_crop = crop(depth_img, cX, cY, output_size)
    rgb_crop = crop(rgb_img, cX, cY, output_size)
    mask_crop = crop(mask, cX, cY, output_size)
    if threshold_object_table is not None:
        threshold_object_table = int(threshold_object_table)
        mask_table = extract_mask_from_img(depth_crop, threshold=threshold_object_table, lower_than_threshold=False)
        mask_table = cv2.bitwise_or(mask_table, cv2.bitwise_not(mask_crop))
        contours, hierarchy = cv2.findContours(depth_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('mask_table', mask_table)
        # depth_color = cv2.cvtColor(depth_crop, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(depth_color, contours, 0, (0, 255, 0), 1)
        # cv2.imshow('depth_color', depth_color)
        # cv2.imshow('depth_crop', depth_crop)
        # cv2.waitKey(0)

        mask_contours = np.zeros_like(mask_crop)
        cv2.drawContours(mask_contours, contours, 0, (255), 1)
        mean_contour_depth = cv2.mean(depth_crop, mask=mask_contours)[0]
        print(mean_contour_depth)
        mask_object = cv2.bitwise_not(mask_table)
        depth_crop = cv2.bitwise_and(depth_crop, depth_crop, mask=mask_object)
        perfect_table = mask_table/255*int(mean_contour_depth)
        perfect_table = perfect_table.astype(np.uint8)
        depth_crop = cv2.bitwise_or(depth_crop, perfect_table)
        
    return rgb_crop, depth_crop, mask_crop
    