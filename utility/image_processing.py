import cv2
import numpy as np
import enum
from dataclasses import dataclass
import torch
from scipy.ndimage import label

class DistortionModel(enum.Enum):
    PINHOLE = 0  # no distortion
    OPENCV = 1  # opencv calibration (distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements.

@dataclass(frozen=True)
class CameraIntrinsics:
    """
    CameraIntrinsics is a container for camera intrinsic parameters.

    Camera intrinsics matrix K = [fx  s cx]
                                 [ 0 fy cy]
                                 [ 0  0  1] parameterized by:
     - fx: focal length in x-direction
     - fy: focal length in y-direction
     - cx: principal point in x-direction
     - cy: principal point in y-direction
     - s: skew factor (very unusual)

    width and height are the resolution of the camera image.
    distortion_model is the model of the distortion.
    distortion_coeffs are the coefficients of the distortion model (none for PINHOLE)
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: DistortionModel = DistortionModel.PINHOLE
    distortion_coefficients: np.ndarray = np.array([])
    s: float = 0.0

    @property
    def intrinsics_matrix(self):
        return np.array([[self.fx, self.s, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def __repr__(self) -> str:
        return f"CameraIntrinsics(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, width={self.width}, height={self.height}, distortion_model={self.distortion_model}, distortion_coefficients={self.distortion_coefficients}, s={self.s})"


def extract_mask_threshold(
    image: np.ndarray, threshold: float, lower_than_threshold: bool = True
) -> np.ndarray:
    """
    Extract a mask from the image using a threshold.
    """
    if image.ndim == 3:
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = image
    if lower_than_threshold:
        mask = im_gray < threshold
    else:
        mask = im_gray > threshold
    return mask

def extract_mask_threshold_torch(
    tensor: torch.Tensor, threshold: float, lower_than_threshold: bool = True
) -> np.ndarray:
    """
    Extract a mask from the tensor using a threshold.
    """
    if tensor.ndimension() == 3:
        im_gray = tensor.mean(dim=0)
    else:
        im_gray = tensor
    if lower_than_threshold:
        mask = im_gray < threshold
    else:
        mask = im_gray > threshold
    return mask

def largest_region_with_scipy(tensor: torch.Tensor) -> torch.Tensor:
    # Convert to numpy
    np_tensor = tensor.cpu().numpy()

    # Label the regions (4-connectivity by default)
    labeled_array, num_features = label(np_tensor)

    # Find the largest region label
    max_region_size = 0
    largest_region_label = 0
    for region_label in range(1, num_features + 1):  # Start from 1 because 0 is the background
        region_size = (labeled_array == region_label).sum()
        if region_size > max_region_size:
            max_region_size = region_size
            largest_region_label = region_label

    # Create a mask for the largest region
    largest_region_mask = (labeled_array == largest_region_label)

    # Convert back to torch tensor
    return torch.from_numpy(largest_region_mask).to(tensor.device)

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
    mask = mask.astype(np.uint8)
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

def crop(img, centerX, centerY, crop_sizeX, crop_sizeY=None):
    """
    Crop an image around a center point.
    Args:
        img: np.array: The image to crop.
        centerX: int: The x coordinate of the center, or the column in matrix notation. 
        cernterY: int: The y coordinate of the center, or the line in in matrix notation.
        crop_sizeX: int: The size of the crop on the X axis (the column indexin matrix notations), or on both sides if crop_sizeY is None.
        crop_sizeY: int: The size of the crop. If None, crop_sizeY will be equal to crop_sizeX.
    Returns:
        np.array: The cropped image.
    """
    if crop_sizeY==None:
        crop_sizeY = crop_sizeX
    shape = img.shape
    left = max(0, min(centerX - crop_sizeX//2, shape[1] - crop_sizeX))
    top = max(0, min(centerY - crop_sizeY//2, shape[0] - crop_sizeY))
    bottom = min(shape[0], top + crop_sizeY)
    right = min(shape[1], left + crop_sizeX)
    return img[top:bottom, left:right]

def process_rgbd_perfect_table(rgb_img, depth_img, mask=None, output_size=480, threshold_object_table=None, scale_factor = 1):
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
    int_mult = 1
    if depth_img.dtype == np.uint8:
        depth_img_int = depth_img.copy()
    else:
        # opencv expects uint8 for most functions so we need to scale the depth image to not lose too much precision
        int_mult = 127
        depth_img_int = depth_img*int_mult
        depth_img_int = depth_img_int.astype(np.uint8)    
    depth_img = depth_img.astype(np.float32)
    if mask is None:
        #! THRESHOLD VALUES MIGHT NEED TO BE CHANGED
        mask = extract_mask_threshold(rgb_img, threshold=1, lower_than_threshold=False)
        mask = extend_mask(mask, extend_by=101, shape_square=False)
    depth_img = cv2.bitwise_or(depth_img, depth_img, mask=mask)

    cX, cY = get_center_of_mass(mask)
    depth_crop = crop(depth_img, cX, cY, output_size)
    depth_crop_int = crop(depth_img_int, cX, cY, output_size)
    rgb_crop = crop(rgb_img, cX, cY, output_size)
    mask_crop = crop(mask, cX, cY, output_size)
    if threshold_object_table is not None:
        threshold_object_table = int(threshold_object_table)
        mask_table = extract_mask_threshold(depth_crop_int, threshold=threshold_object_table, lower_than_threshold=False)
        # print(mask_table.dtype)
        # print(mask_crop.dtype)
        # print(mask_table.shape)
        # print(mask_crop.shape)
        mask_table = cv2.bitwise_or(mask_table, cv2.bitwise_not(mask_crop))
        contours, hierarchy = cv2.findContours(depth_crop_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('mask_table', mask_table)
        # depth_color = cv2.cvtColor(depth_crop, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(depth_color, contours, 0, (0, 255, 0), 1)
        # cv2.imshow('depth_color', depth_color)
        # cv2.imshow('depth_crop', depth_crop)
        # cv2.waitKey(0)

        mask_contours = np.zeros_like(mask_crop)
        cv2.drawContours(mask_contours, contours, 0, (255), 1)
        mean_contour_depth = cv2.mean(depth_crop_int, mask=mask_contours)[0]
        mean_contour_depth = mean_contour_depth/int_mult
        print(mean_contour_depth)
        mask_object = cv2.bitwise_not(mask_table)
        depth_crop = cv2.bitwise_and(depth_crop, depth_crop, mask=mask_object)
        # perfect_table = np.zeros_like(mask_table, dtype=np.float32)
        perfect_table = np.array(mask_table,dtype=np.float32)/255*mean_contour_depth
        print(perfect_table.dtype)
        print(perfect_table.shape)
        print(perfect_table.max())
        print(perfect_table.min())
        print(perfect_table.mean())
        print((np.array(mask_table,dtype=np.float32)/255*mean_contour_depth).max())
        print(np.array(mask_table,dtype=np.float32).min())
        print(np.array(mask_table,dtype=np.float32).mean())
        print(np.array(mask_table,dtype=np.float32).dtype)
        cv2.imshow('perfect_table', perfect_table)
        cv2.imshow('mask_table', mask_table)
        cv2.imshow('perfect_table_int', perfect_table.astype(np.uint8))
        cv2.waitKey(0)
        
        depth_crop = cv2.bitwise_or(depth_crop, perfect_table)
        
    return rgb_crop, depth_crop, mask_crop
    
def process_rgbd_inpaint(rgb_img, depth_img, mask=None, output_size=480, threshold_object_table=None, scale_factor = 1):
    """
    Process an RGBD image.
    Args:
        rgb_img: np.array: The RGB image.
        depth_img: np.array: The depth image.
        mask: np.array: The mask to apply to the depth image.
        output_size: int: The size of the output image.
        threshold_object_table: int: Kept for compatibility with process_rgbd_perfect_table.
    """
    if depth_img.ndim == 3:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    depth_nan = np.isnan(depth_img)
    depth_img[depth_nan] = 0 
    depth_img = depth_img.astype(np.float32)
    if mask is None:
        #! THRESHOLD VALUES MIGHT NEED TO BE CHANGED
        mask = extract_mask_threshold(rgb_img, threshold=1, lower_than_threshold=False)
        mask = extend_mask(mask, extend_by=91, shape_square=False)
    depth_img = cv2.bitwise_or(depth_img, depth_img, mask=mask)

    cX, cY = get_center_of_mass(mask)
    depth_crop = crop(depth_img, cX, cY, output_size)
    rgb_crop = crop(rgb_img, cX, cY, output_size)
    mask_crop = crop(mask, cX, cY, output_size)
    depth_crop = cv2.inpaint(depth_crop, cv2.bitwise_not(mask_crop), 3, cv2.INPAINT_TELEA)
    
    cX, cY = get_center_of_mass(mask_crop)
    if scale_factor != 1:
        rgb_zoom = zoom_image(rgb_crop, scale_factor, cX, cY)
        depth_zoom = zoom_image(depth_crop, scale_factor, cX, cY)
        mask_zoom = zoom_image(mask_crop, scale_factor, cX, cY)
        return rgb_zoom, depth_zoom, mask_zoom
    else:
        return rgb_crop, depth_crop, mask_crop
    
def zoom_image(img, factor, cX=None, cY=None):
    """
    Zoom or dezoom an image.
    Args:
        img: np.array: The image to rescale.
        factor: float: The factor to apply, between 0 and 1 for dezooming, and above 1 for zooming.
    Returns:
        np.array: The zoomed image, with the same shape as the input image.
    """
    assert factor>0, "Factor must be positive"
    base_shape = img.shape
    new_shape = np.array(base_shape[:-1])*factor
    new_shape = new_shape.astype(int)
    if new_shape[0]%2 == 1:
        new_shape[0] += 1 # Ensure the shape is even
        new_shape[1] += 1
    if cX is None:
        cX = new_shape[1]//2 
    else:
        cX = int(cX*factor)
    if cY is None:
        cY = new_shape[0]//2
    else:
        cY = int(cY*factor)
    if factor>1:
        img_zoomed = cv2.resize(img, (new_shape[1],new_shape[0]))
        return crop(img_zoomed, cX, cY, base_shape[1], base_shape[0])
    elif factor<1:
        img_dezoomed = cv2.resize(img, (new_shape[1],new_shape[0]))
        border_sizeX = (base_shape[1] - new_shape[1] )//2
        border_sizeY = (base_shape[0] - new_shape[0] )//2
        img_dezoomed = cv2.copyMakeBorder(img_dezoomed, border_sizeY, border_sizeY, border_sizeX, border_sizeX, cv2.BORDER_REPLICATE)
        return img_dezoomed
    return img