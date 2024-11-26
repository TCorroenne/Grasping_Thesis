import os
import glob
import numpy as np
import random
import re
import cv2
from .grasp_data import GraspDatasetBase
from ..grasp import GraspRectangles
from ..image import DepthImage, Image


class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, image_wise=False, random_seed=10, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        if image_wise:
            random.seed(random_seed)
            random.shuffle(graspf)
        else:
            graspf.sort()
        l = len(graspf)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]
        # print("Number of images: ", len(self.grasp_files))

    def _get_crop_attrs(self, idx):
        gtbbs = GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        # print("{},{},{},{}".format(depth_img.max(), depth_img.min(), np.mean(depth_img), np.std(depth_img)))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            # print("{},{},{},{}".format(rgb_img.max(), rgb_img.min(), np.mean(rgb_img), np.std(rgb_img)))
        return rgb_img.img


class CustomDataset(GraspDatasetBase):
    """
    custom dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, image_wise=False, random_seed=10, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CustomDataset, self).__init__(**kwargs)

        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos_s*.txt'))
        if image_wise:
            random.seed(random_seed)
            random.shuffle(graspf)
        else:
            graspf.sort()
        l = len(graspf)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = []
        rgbf = []
        scoresf = []
        angles_at_faultf = []
        for f in graspf:
            match = re.search(r"cpos_s", f)
            start_match = match.start()
            end_match = match.end()
            depthf.append(f[:start_match] + "d.tiff")
            rgbf.append(f[:start_match] + "r.png")
            scoresf.append(int(f[end_match]))
            angles_at_faultf.append(f[end_match+1:end_match+4] == "ang")
            # depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
            # rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]
        self.scores = scoresf[int(l*start):int(l*end)]
        self.angles_at_fault = angles_at_faultf[int(l*start):int(l*end)]
        # print("Number of images: ", len(self.grasp_files))

    def _get_crop_attrs(self, idx):
        gtbbs = GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 480 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0, symmetry=0, center_zoom=None):
        gtbbs = GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.offset((-top, -left))
        if center_zoom is None:
            center_zoom = (self.output_size//2, self.output_size//2)
        gtbbs.zoom_croping(zoom, center_zoom)
        gtbbs.rotate(rot, (self.output_size//2, self.output_size//2))
        gtbbs.symmetry(symmetry, self.output_size)
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0, symmetry=0, get_center_zoom=False):
        depth_img = DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(480, left + self.output_size)))
        depth_img.normalise()
        center_zoom = depth_img.zoom(zoom, center=center)
        depth_img.resize((self.output_size, self.output_size))
        depth_img.rotate(rot)
        depth_img.symmetry(symmetry)
        # print("{},{},{},{}".format(depth_img.max(), depth_img.min(), np.mean(depth_img), np.std(depth_img)))
        if get_center_zoom:
            return depth_img.img, center_zoom
        else:
            return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True, brightness=0, contrast=1.0, symmetry=0):
        rgb_img = Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(480, left + self.output_size)))
        rgb_img.zoom(zoom, center=center)
        rgb_img.resize((self.output_size, self.output_size))
        rgb_img.img = cv2.convertScaleAbs(rgb_img.img, alpha=contrast, beta=brightness)
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            # print("{},{},{},{}".format(rgb_img.max(), rgb_img.min(), np.mean(rgb_img), np.std(rgb_img)))
        rgb_img.rotate(rot)
        rgb_img.symmetry(symmetry)
        return rgb_img.img

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.75, 1.0)
        else:
            zoom_factor = 1.0

        if self.random_symmetry and self.random_rotate:
            symmetries = [0, 1] # because we rotate the image, we get all possible combinaisons with only 2 symmetries
            sym = random.choice(symmetries)
        elif self.random_symmetry:
            symmetries = [0, 1, 2, 3]
            sym = random.choice(symmetries)
        else:
            sym = 0
        
        if self.random_brightness:
            brightness = np.random.randint(-10,11)
        else:
            brightness = 0
        
        if self.random_contrast:
            contrast = np.random.uniform(0.91, 1.1)
        else:
            contrast = 1.0
        
        # Load the depth image
        if self.include_depth:
            depth_img, center_zoom = self.get_depth(idx, rot, zoom_factor, symmetry=sym, get_center_zoom=True)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor, brightness=brightness, contrast=contrast, symmetry=sym)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor, symmetry=sym, center_zoom=center_zoom)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.max_width)/self.max_width

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor, self.scores[idx], self.angles_at_fault[idx]