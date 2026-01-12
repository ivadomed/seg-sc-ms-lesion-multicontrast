"""
This script adds  fakes lesions to healthy spinal cord MRI images using predefined lesion shapes.
It takes as 
"""

from pathlib import Path
import os
import random
from typing import Tuple, Optional, Union, List, Generator
import argparse
import scipy.stats
import torch
import torchio as tio
import skimage.morphology as morph
import numpy as np
from scipy.ndimage import center_of_mass
from copy import deepcopy
import warnings
from matplotlib import pyplot as plt


def get_bbox_bounds(im):
    """ Returns the min and max indices of the bounding box of non-zero voxels in im. """
    ax0 = np.any(im, axis=(1, 2))
    ax1 = np.any(im, axis=(0, 2))
    ax2 = np.any(im, axis=(0, 1))
    ax0_min, ax0_max = np.where(ax0)[0][[0, -1]]
    ax1_min, ax1_max = np.where(ax1)[0][[0, -1]]
    ax2_min, ax2_max = np.where(ax2)[0][[0, -1]]
    bounds = [(ax0_min, ax0_max), (ax1_min, ax1_max), (ax2_min, ax2_max)]
    # Convert to int (rather than numpy.int64) to avoid errors when saving to json
    bounds = [(int(min_), int(max_)) for min_, max_ in bounds]
    return bounds


def gaussian_func(x: Union[np.ndarray, torch.Tensor], sigma: Union[float, np.ndarray, List[float]],
                  amplitude: float = 1.0, norm=False) -> np.ndarray:
    """Return the m-dimensional Gaussian function/kernel at the given points.
    Args:
        x: The points at which to evaluate the Gaussian function.
        sigma: The standard deviation of the Gaussian function. If one value is given, the same sigma is used for all
               dimensions. A list of m values can be given to use different sigmas for each dimension.
        amplitude: The amplitude of the Gaussian function, i.e. the max value at the centre of the Gaussian.
        norm: If True, the kernel is normalised to sum to 1.
    Returns:
        The Gaussian function evaluated at the given points. The amplitude value is reached at the origin.
    """
    # Calculate the Gaussian function
    m = x.shape[-1] if x.ndim > 1 else 1  # Number of dimensions
    if isinstance(sigma, list):
        assert len(sigma) == m, 'If input sigma is a list, then the length must match the number of dimensions in x'

    if isinstance(x, np.ndarray):
        if m > 1:
            z = amplitude * np.exp(-np.sum((x / sigma)**2, axis=-1) / 2)
        else:
            z = amplitude * np.exp(-x**2 / (2 * sigma**2))
    elif isinstance(x, torch.Tensor):
        if m > 1:
            z = amplitude * torch.exp(-torch.sum((x / sigma)**2, dim=-1) / 2)
        else:
            z = amplitude * torch.exp(-x**2 / (2 * sigma**2))
    else:
        raise ValueError('Input x must be a numpy array or a torch tensor')

    if norm:
        z /= z.sum()

    return z


def find_seg_files(lesion_dir) -> List:
    return [
        Path(root) / file
        for root, _, files in os.walk(lesion_dir, followlinks=True)
        for file in files
        if file.endswith('.nii.gz') and 'seg' in file
    ]


class LesionSCynth(tio.Transform):
    r"""Randomly add predefined lesion shapes by increasing contrast.
    Args:
        factor_distribution: a scipy.stats.rv_continuous distribution to sample the intensity factor from.
        modalities: a list of modality names to add the lesions to, corresponding to the keys in the tio subjects.
        lesion_dir: a directory containing the binary lesion masks to add.
                    If lesion_dir is provided, the lesion masks should be saved as '.nii.gz' and should contain 'seg'.
                    Either lesion_dir or lesion_paths must be provided.
        lesion_paths: a list of paths to the lesion masks to add. Either lesion_dir or lesion_paths must be provided.
        min_size: an optional minimum size for the lesion masks to use.
        max_size: an optional maximum size for the lesion masks to use.
        dilate_erode_probability: the probability of dilating or eroding the lesion mask. Default: 0.5.
        other_transforms: an optional tio.Transform to apply to the lesion mask.
        multimodality_probability: the probability of adding the lesion to more than one modality. Default: 0.5.
                                   Only applies if more than one modality is provided.
        gaussian_spatial: if True, intensity increase will be modelled spatially by a Gaussian function. Default: False.
        min_factor_gaussian: the minimum increase factor to use for the Gaussian intensity increase. Default: 0.0.
        gaussian_sigma: a fixed standard deviation of the Gaussian function to model the intensity increase spatially.
                        Default: None - Sampled randomly for each dimension, based on the length of the lesion.
        blur_radius: the radius of the Gaussian blur to apply to the input image in the lesion area. Default: None.
                     if None, then no blur is applied.
        blur_sigma: the sigma of the Gaussian blur to apply to the input image in the lesion area. Default: None.
        n_lesions_dist: an optional generator to sample the number of lesions to add. Should be iterable with next()
                        and should return an integer. Default: None. By default, a random number between 0 and 8 is
                        chosen.
        fixed_synthetic_lesions: if True, the same lesions will be added to the same subject each time. Default: False.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """
    def __init__(
            self,
            factor_distribution: scipy.stats.rv_continuous,
            modalities: List[str],
            lesion_dir: Optional[Path] = None,
            lesion_paths: Optional[list] = None,
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
            dilate_erode_probability: float = 0.5,
            other_transforms: Optional[tio.Transform] = None,
            multimodality_probability: float = np.sqrt(0.5),
            gaussian_spatial: bool = False,
            min_factor_gaussian: float = 0.0,
            gaussian_sigma: Optional[float] = None,
            blur_radius: Optional[Union[int, Tuple[int, int, int]]] = None,
            blur_sigma: Optional[Union[float, Tuple[float, float, float]]] = None,
            n_lesions_dist: Optional[Generator] = None,
            fixed_synthetic_lesions: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert (lesion_dir is not None) != (lesion_paths is not None), 'One of lesion_dir or lesion_paths must be provided'
        # Get a list of all possible lesions
        if lesion_dir is not None:
            # rglob doesn't work if subdirs are symlinks
            # self.lesion_paths = [f for f in lesion_dir.rglob('*.nii.gz') if 'seg' in f.name]
            self.lesion_paths = find_seg_files(lesion_dir)
        else:
            self.lesion_paths = lesion_paths

        self.factor_dist = factor_distribution
        self.modalities = modalities
        self.n_lesions_dist = n_lesions_dist

        self.dilate_erode_probability = dilate_erode_probability
        self.other_transforms = other_transforms
        self.multimodality_probability = multimodality_probability

        self.gaussian_spatial = gaussian_spatial
        self.min_factor_gaussian = min_factor_gaussian
        self.gaussian_sigma = gaussian_sigma
        assert (blur_sigma is not None) == (blur_radius is not None), \
            'Both blur_sigma and blur_radius must be provided or neither.'
        self.blur_radius = blur_radius
        self.blur_sigma = blur_sigma

        self.min_size = min_size
        self.max_size = max_size

        self.fixed_synthetic_lesions = fixed_synthetic_lesions

    def get_lesion(self) -> Optional[tio.LabelMap]:
        """ Get a random lesion mask from the available lesions. If min_size or max_size are not None, ensure that the
        lesion is within the specified size range. If no lesions are found, return None.
        """
        max_iter = 10

        if self.min_size is None and self.max_size is None:
            lesion_path = random.choice(self.lesion_paths)
            return tio.LabelMap(lesion_path)
        else:
            for _ in range(max_iter):
                lesion_path = random.choice(self.lesion_paths)
                lesion_im = tio.LabelMap(lesion_path)
                if self.min_size <= lesion_im.data.sum() <= self.max_size:
                    return lesion_im

            msg = (f'Could not find a lesion with size between {self.min_size} and {self.max_size} after '
                   f'{max_iter} iterations. ')
            warnings.warn(msg)

    def model_lesion_gaussian(self, mask: tio.LabelMap) -> Tuple[np.ndarray, np.ndarray]:
        """ Based on a given mask shape, use a Gaussian function to model the intensity increase.
        The Gaussian will be maximal (1.0) at the centroid of the mask and will decay towards the edges. 
        The speed of decay is controlled by sigma and is sampled for each dimension from a uniform distribution between
        1/8 and 1 times the extent of the mask in each dimension. 
        Args:
            mask: a torchio.LabelMap instance containing the binary lesion mask to use.
        Returns:
            gaussians: a numpy array containing the Gaussian function evaluated at all points in the mask image.
            sigma: the standard deviation(s) used in the Gaussian function.
        """
        bounds = get_bbox_bounds(mask.data[0].numpy())
        # Get the length/extent of the mask in each dimension
        extent = np.array([max_bound - min_bound + 1 for (min_bound, max_bound) in bounds])

        CoM = center_of_mass(mask.data[0].numpy())  # (x, y, z)
        # Get location of each point relative to CoM
        shp = mask.spatial_shape
        # Create a grid of points in the mask image. The following gives (xx, yy, zz), a tuple of
        grid = np.meshgrid(np.arange(shp[0]), np.arange(shp[1]), np.arange(shp[2]), indexing='ij')
        grid = np.stack(grid, axis=-1)
        dist = np.array(CoM) - grid

        if self.gaussian_sigma is None:
            sigma = np.array([scipy.stats.uniform(loc=e / 8, scale=e).rvs() for e in extent])
        else:
            sigma = self.gaussian_sigma
        gaussians = gaussian_func(dist, sigma, amplitude=1.0)

        return gaussians, sigma

    def dilate_erode_mask(self, mask_data: torch.Tensor) -> torch.Tensor:
        assert mask_data.ndim == 4, 'Mask data must be 4D, with shape (1, H, W, D) with a batch dimension of 1'
        rand = random.random()
        if rand < self.dilate_erode_probability:
            structuring_element = morph.ball(1)
            # Half of the time erode and half of the time dilate
            if rand < self.dilate_erode_probability / 2:
                mask_data = torch.Tensor(morph.binary_dilation(mask_data[0], footprint=structuring_element)).unsqueeze(0)
            else:
                mask_data = torch.Tensor(morph.binary_erosion(mask_data[0], footprint=structuring_element)).unsqueeze(0)
        return mask_data

    def apply_blur(self, inp: torch.Tensor, seg: torch.Tensor, sc_seg: torch.Tensor) -> torch.Tensor:
        if self.blur_radius is None:
            return inp

        # Dilate the segmentation mask
        structuring_element = morph.ball(2)
        dilated_seg = torch.Tensor(morph.binary_dilation(seg[0], footprint=structuring_element)).unsqueeze(0)

        # Apply the blur to the input image
        blurred = scipy.ndimage.gaussian_filter(inp, sigma=self.blur_sigma, radius=self.blur_radius)

        # Only keep the blurred values within the dilated segmentation mask
        new_inp =  inp * (1 - dilated_seg) + torch.Tensor(blurred) * dilated_seg * sc_seg

        # Now for all values equal to 0, we replace them by their previous value (to avoid black holes)
        new_inp[new_inp == 0] = inp[new_inp == 0]

        return new_inp

    def get_intensity_increase(self, lesion_im: tio.LabelMap) -> Tuple[torch.Tensor, float, Optional[float]]:
        """ Get the intensity increase factor for the lesion. If self.gaussian_spatial is True, the intensity increase
        will be modelled spatially by a Gaussian function. If False, the intensity increase will be constant across the
        lesion mask.
        Args:
            lesion_im: a torchio.LabelMap instance containing the binary lesion mask where the lesion will be created.
        Returns:
            intensity_increase: a torch.Tensor containing the Gaussian function evaluated at all points in the mask
                                image or a constant value.
            factor: the intensity increase factor.
            sigma: the standard deviation(s) used in the Gaussian function.
        """
        # Choose a random intensity factor
        factor = self.factor_dist.rvs()
        if self.gaussian_spatial:
            gaussian, sigma = self.model_lesion_gaussian(lesion_im)
            gaussian = torch.Tensor(gaussian)
            # Calculate the average intensity increase with amplitude=1.0, and then scale it to achieve the desired
            # average intensity increase factor.
            avg_factor = (gaussian * lesion_im.data).sum() / lesion_im.data.sum()
            amplitude = factor / avg_factor
            intensity_increase = torch.Tensor(amplitude * gaussian)
            # Set a contant minimum factor, since the Gaussian can decay close to zero at the edges, and we want to
            # maintain some intensity increase everywhere within the mask
            intensity_increase[intensity_increase < self.min_factor_gaussian] = self.min_factor_gaussian
        else:
            intensity_increase = torch.Tensor([factor])
            sigma = None
        return intensity_increase, factor, sigma

    @staticmethod
    def update_intensity(subject, lesion_im, intensity_increase, modality_name, z):
        """
        Update the intensity of the input image by increasing the intensity within the lesion mask area.
        Args:
            subject: tio.Subject instance containing the intensity image to augment, and the spinal cord
                     segmentation (named 'sc_seg').
            lesion_im: a torchio.LabelMap instance containing the binary lesion mask to use, cropped to the bounding
                       box of the lesion along the z-axis.
            intensity_increase: a torch.Tensor used for the multiplicative intensity increase, containing either:
                                1) a constant value for the entire lesion mask, or
                                2) a Gaussian function (or other spatially varying function) evaluated at all points
                                in the mask image, with same shape as lesion_im.
            modality_name: the name of the modality in subject to augment. e.g. 'image' or 't2', etc.
            z: the z-location where the lesion will be inserted into the input image in subject. This is the lower
               bound of the new lesion.
        Returns:
            subject: the input subject with the intensity increased within the lesion mask area.
        """
        # Add the intensity increase to the relevant part of the image.
        # 1. The increase is multiplied by the existing intensities.
        # 2. The increase is multiplied by the lesion mask to ensure that the intensity is only increased within the
        #    target lesion mask area.
        # 3. The increase is multiplied by the spinal cord segmentation to ensure that the intensity is only increased
        #    within the spinal cord area.
        subject[modality_name].data[..., z:z+lesion_im.shape[-1]] += (
            subject[modality_name].data[..., z:z+lesion_im.shape[-1]]
            * intensity_increase
            * lesion_im.data
            * subject['sc_seg'].data[..., z:z+lesion_im.shape[-1]]  # Only add lesion to the SC
        )
        return subject

    @staticmethod
    def get_target_position(subject: tio.Subject, lesion_im: tio.LabelMap) -> int:
        """ Get the target z-location for the lesion to be inserted into the input image. Currently just a
        random position along the z-axis, but could be improved to use a prior probability map. The z index determines
        the lower bound of the inserted lesion.
        Args:
            subject: a torchio.Subject instance containing the image (self.modality_name) and
                     segmentation mask ('segmentation') to augment.
            lesion_im: a torchio.LabelMap instance containing the binary lesion mask to use, cropped to the bounding
                       box of the lesion along the z-axis.
        Returns:
            z: the z index where the lesion will be inserted into the input image in subject.
        """
        while True:
            z = random.randint(0, subject.spatial_shape[-1] - lesion_im.shape[-1])
            # Ensure that there is some spinal cord in this region
            if subject.sc_seg.data[..., z:z+lesion_im.shape[-1]].sum() > 0:
                break
        return z

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Apply the LesionSCynth augmentation for a single subject by inserting several lesions into the input image.
        Args:
            subject: a torchio.Subject instance containing the image (self.modality_name) and
                     segmentation mask ('segmentation') to augment.
        Returns:
            subject: a torchio.Subject instance with the augmented image and segmentation mask.
        """
        lower_n_lesions, upper_n_lesions = 0, 8
        if self.fixed_synthetic_lesions:
            random.seed(int(subject['name']))
            np.random.seed(int(subject['name']))  # used by scipy.stats
            # For a small fixed subset of synthetic lesion augmentations, we ensure that we always insert lesions
            lower_n_lesions, upper_n_lesions = 1, 8
        # Pick a random number of lesions to add
        if self.n_lesions_dist is not None:
            n_lesions = next(self.n_lesions_dist)
        else:
            n_lesions = random.randint(lower_n_lesions, upper_n_lesions)
        
        # In this case, we only add 1 lesion:
        n_lesions = 1

        # Reorient the image to RAS
        subject.image = tio.ToOrientation('RAS')(subject.image)
        subject.sc_seg = tio.ToOrientation('RAS')(subject.sc_seg)

        for _ in range(n_lesions):
            print(subject.image)
            lesion_im = self.get_lesion()
            print(lesion_im)
            ### I needed to perform some preprocessing here
            # Reorient the lesion to match the subject
            img_orientation = subject.image.orientation
            lesion_im = tio.ToOrientation(''.join(img_orientation))(lesion_im)
            # Re-sample the lesion to match the subject resolution
            img_resolution = subject.image.spacing
            lesion_im = tio.Resample(img_resolution, image_interpolation='nearest')(lesion_im)
            print(lesion_im)
            # Crop the mask to only keep the box containing the lesion
            bounds = get_bbox_bounds(lesion_im.data[0].numpy())
            _, X, Y, Z = lesion_im.shape
            x_l, x_r = bounds[0][0], X - 1 -bounds[0][1]
            y_l, y_r = bounds[1][0], Y - 1 - bounds[1][1]
            z_l, z_r = bounds[2][0], Z - 1 - bounds[2][1]
            lesion_im = tio.Crop(cropping=(x_l, x_r, y_l, y_r, z_l, z_r))(lesion_im)

            if lesion_im is None:
                continue

            # Choose a random z-location for the lesion
            z = self.get_target_position(subject, lesion_im)
            print(f'Inserting lesion at z={z} with shape {lesion_im.shape}')

            # Now we create an empty label with the XY shape of the image and the Z shape of the lesion
            lesion_im_padded = deepcopy(lesion_im)
            lesion_im_padded.data = torch.zeros((1, subject.spatial_shape[0], subject.spatial_shape[1], lesion_im.shape[-1]))
            # We found the center of the spinal cord in XY in the Z chunk of the lesion
            sc_center = center_of_mass(subject.sc_seg.data[..., z:z+lesion_im.shape[-1]].numpy())
            print("sc_center", sc_center)

            # We now compute the bounds of the sc in the chunk:
            sc_chunk = subject.sc_seg.data[0, ..., z:z+lesion_im.shape[-1]].numpy()
            bounds = get_bbox_bounds(sc_chunk)
            sc_x_width = bounds[0][1] - bounds[0][0] + 1
            sc_y_width = bounds[1][1] - bounds[1][0] + 1
            print(f"sc_x_width: {sc_x_width}, sc_y_width: {sc_y_width}")

            # Now we find the lesion widths in X and Y
            lesion_bounds = get_bbox_bounds(lesion_im.data[0].numpy())
            lesion_x_width = lesion_bounds[0][1] - lesion_bounds[0][0] + 1
            lesion_y_width = lesion_bounds[1][1] - lesion_bounds[1][0] + 1
            print(f"lesion_x_width: {lesion_x_width}, lesion_y_width: {lesion_y_width}")

            # The lesion is centered on the spinal cord with an offset in X and Y so that 0.5 lesion_width + offset <= 0.5 sc_width
            max_x_offset = max(0, (sc_x_width - lesion_x_width) // 2)
            max_y_offset = max(0, (sc_y_width - lesion_y_width) // 2)
            x_offset = random.randint(-max_x_offset, max_x_offset)
            y_offset = random.randint(-max_y_offset, max_y_offset)
            print(f"x_offset: {x_offset}, y_offset: {y_offset}")
            lesion_center = np.copy(sc_center)
            lesion_center[1] += x_offset
            lesion_center[2] += y_offset
            # Round lesion center to nearest integer
            lesion_center = [int(round(c)) for c in lesion_center]
            print("lesion_center", lesion_center)

            # Now we compute the lesion center for the cropped lesion
            lesion_center_ini = center_of_mass(lesion_im.data.numpy())
            lesion_center_ini = [int(round(c)) for c in lesion_center_ini]
            print("lesion_center_ini", lesion_center_ini)

            # Compute lesion displacement:
            displacement_x = lesion_center[1] - lesion_center_ini[1]
            displacement_y = lesion_center[2] - lesion_center_ini[2]

            # Now we copy the lesion into the padded lesion tensor
            lesion_im_padded.data[
                :,
                displacement_x:displacement_x + lesion_im.shape[1],
                displacement_y:displacement_y + lesion_im.shape[2],
                :
            ] = lesion_im.data
            print(lesion_im_padded)
            lesion_im = lesion_im_padded

            if len(self.modalities) == 1:
                lesion_im.set_data(self.dilate_erode_mask(lesion_im.data))
                # Apply other transforms if provided (e.g. rotate, scale, etc.)
                if self.other_transforms is not None:
                    lesion_im = self.other_transforms(lesion_im)

                if lesion_im.data.sum() == 0:
                    continue

                # Get Gaussian or constant intensity kernel
                intensity_increase, factor, sigma = self.get_intensity_increase(lesion_im)
                modality = self.modalities[0]
                # Increase the intensity by the Gaussian within the lesion mask area
                subject = self.update_intensity(subject, lesion_im, intensity_increase, modality, z)
                # Add the lesion to the segmentation mask
                subject['segmentation'].data[..., z:z+lesion_im.shape[-1]] = deepcopy(lesion_im.data)

        # Ensure to remove any parts of mask outside spinal cord (as we have not increased the intensity)
        subject['segmentation'].set_data(subject['segmentation'].data * subject['sc_seg'].data)
        print("segmentation", subject.segmentation)

        if self.blur_radius is not None:
            for modality in self.modalities:
                subject[modality].set_data(self.apply_blur(subject[modality].data, subject['segmentation'].data, subject['sc_seg'].data))

        return subject


class OptionalLesionSCynth(LesionSCynth):
    @staticmethod
    def check_condition(subject: tio.Subject) -> bool:
        # Only insert lesions into acquisitions with no lesions (i.e. where label == 0)
        return subject['label'] == 0

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.check_condition(subject):
            return super().apply_transform(subject)
        return subject


class CarveMix(tio.Transform):
    """ A re-implementation of the simple CarveMix method from:
    X. Zhang et al., ‘CarveMix: A simple data augmentation method for brain lesion segmentation’,
    NeuroImage, vol. 271, p. 120041, May 2023, doi: 10.1016/j.neuroimage.2023.120041.
    The official implementation is available at: https://github.com/ZhangxinruBIT/CarveMix

    In this implementation, which deviates from the original paper slightly, we take a volume which has no lesions
    and add the lesions from another volume to it. The exact area that is taken around each lesion is determined by
    comparing the distance to the lesion boundary to a threshold sampled from a probability distribution (as in the
    original paper).
    Args:
        lesion_subjects: a list of torchio.Subject instances containing the lesion images and masks to add.
        im_name: the name of the anat image in the torchio.Subject instance into which the lesion is inserted.
        seg_name: the name of the segmentation mask in the torchio.Subject instance into which the lesion is inserted.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """
    def __init__(
            self,
            lesion_subjects: List[tio.Subject],
            im_name: str = 'image',
            seg_name: str = 'segmentation',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.lesion_subjects = lesion_subjects
        self.im_name = im_name
        self.seg_name = seg_name

    @staticmethod
    def check_condition(subject: tio.Subject) -> bool:
        # Only insert lesions into acquisitions with no lesions (i.e. where label == 0)
        return subject['label'] == 0

    def get_lesion_im(self) -> tio.Subject:
        # Select a random lesion volume to insert, and return the image and lesion mask
        return random.choice(self.lesion_subjects)

    @staticmethod
    def get_signed_distances(mask: torch.Tensor, spacing: Optional[Union[List, Tuple]] = None) -> torch.Tensor:
        """  Get the signed distance map for a binary mask, i.e., the distance to the nearest lesion boundary.
        Distance will be positive for voxels outside the lesion, and negative for voxels inside the lesion.
        Args:
            mask: a binary mask with shape (H, W, D) where 1 indicates lesion and 0 indicates background.
            spacing: the spacing of the image. If None, the spacing is assumed to be isotropic with a value of 1.
        Returns:
            A signed distance map with the same shape as the input mask.
        """
        # distance to nearest foreground voxel - will be zero for voxels inside the lesion
        dist_foreground = scipy.ndimage.distance_transform_edt(mask, sampling=spacing)
        # distance to nearest background voxel - will be zero for voxels outside the lesion
        dist_background = scipy.ndimage.distance_transform_edt(1 - mask, sampling=spacing)
        # Want a negative distance for voxels inside the lesion, and a positive distance for voxels outside the lesion
        return dist_background - dist_foreground

    @staticmethod
    def sample_lambda(d: float) -> float:
        """ Sample a threshold to apply to the signed distance map to determine the area around the lesion to extract
        Args:
            d: a positive float representing the maximum distance to the lesion boundary from inside a lesion
        Returns:
            A float representing the threshold to apply to the signed distance map
        """
        assert d > 0, 'The distance d should be positive'

        # Sample delta from Ber(0.5)
        delta = random.choice([0, 1])

        unif_sample = random.random()  # Unif(0, 1)
        if delta:
            return unif_sample * -d/2
        else:
            return unif_sample * d

    @staticmethod
    def norm_combine_unnorm(im1: torch.Tensor, im2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Combine two images by 1) Standardising each image to mean=0 and std=1, 2) Combining the images, and 3)
        Re-scaling the combined image to the original mean and std of the first image.
        """
        # Standardise each image
        im1_mean = im1.mean()
        im1_std = im1.std()
        im1 = (im1 - im1_mean) / im1_std
        im2_mean = im2.mean()
        im2_std = im2.std()
        im2 = (im2 - im2_mean) / im2_std

        # Combine the images
        combined = im1 * (1 - mask) + im2 * mask

        # Un-normalise the combined image
        combined = combined * im1_std + im1_mean

        return combined

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """ Apply the CarveMix augmentation by combining a base subject with parts of another subject volume.
        The inserted subject with lesions will be chosen randomly from the available subjects with lesions.
        Args:
            subject: a torchio.Subject instance containing the image and segmentation mask to augment.
        Returns:
            subject: a torchio.Subject instance with the augmented image and segmentation mask.
        """
        if not self.check_condition(subject):
            return subject

        # Get the lesion image and mask to insert
        lesion_subject = self.get_lesion_im()
        # Ensure the lesion image is the same size as the base image to augment
        lesion_subject = tio.CropOrPad(target_shape=subject.spatial_shape)(lesion_subject)
        lesion_im = lesion_subject[self.im_name].data
        lesion_mask = lesion_subject[self.seg_name].data
        # Get the signed distance map for the lesion mask
        signed_distances = self.get_signed_distances(lesion_mask[0].numpy(), spacing=lesion_subject.spacing)

        # Get the min signed dist (i.e. the max dist from inside the lesion, as distances are negative inside the lesion)
        d = abs(signed_distances.min())

        # Sample a threshold to apply to the signed distance map
        lam = self.sample_lambda(d)
        mask = (signed_distances < lam).astype(np.float32)
        # Insert the masked area of the lesion image into the base subject
        subject[self.im_name].set_data(self.norm_combine_unnorm(subject[self.im_name].data, lesion_im, mask))
        subject[self.seg_name].set_data(subject[self.seg_name].data * (1-mask) + lesion_mask * mask)

        return subject


class LesionMixPopulate(tio.Transform):
    """ A re-implementation of the lesion population branch of LesionMix. Note that the inpainting branch is not
    implemented, and nor is the use of a prior probability map for lesion placement.
    To use this method, the real lesions in the dataset should already be extracted (i.e., cropped to their bounding
    box and saved in separate Nifti files).
    B. D. Basaran, W. Zhang, M. Qiao, B. Kainz, P. M. Matthews, and W. Bai,
    ‘LesionMix: A Lesion-Level Data Augmentation Method for Medical Image Segmentation’,
    in Data Augmentation, Labelling, and Imperfections, MICCAI Workshop 2023.
    Args:
        lesion_dir: a directory containing the cropped lesion masks to add. Should be stored in directories corresponding
                    to subjects/acquisitions. Either lesion_dir or lesion_paths must be provided. If using lesion_dir,
                    then the filenames of the lesion masks should end in '_seg.nii.gz' and the filenames of the
                    corresponding images should end in '.nii.gz'.
        lesion_paths: a list of full paths to the cropped lesion masks to add. The files should be stored in directories
                    corresponding to subjects/acquisitions. The parent dir name of the lesion file will be used to group
                    and sum the volume of the lesions to compute the lesion load and initialise the load distribution
                    from which a target lesion load will be sampled. Each lesion should have a tuple of paths
                    (mask path and intensity image path) with the following structure:
                    [({path_to_acq_dir}/{name_of_mask_file1}.nii.gz, {path_to_acq_dir}/{name_of_image_file1}.nii.gz),
                    ({path_to_acq_dir}/{name_of_mask_file2}.nii.gz, {path_to_acq_dir}/{name_of_image_file2}.nii.gz), ..]
                    Either lesion_dir or lesion_paths must be provided.
        load_distribution: a scipy.stats distribution to sample the lesion load from. Either load_distribution or
                            load_distribution_type must be provided.
        load_distribution_type: a string representing the type of distribution to sample the lesion load from. Either
                            load_distribution or load_distribution_type must be provided. The real lesion load for each
                            subject/acquisition is first computed using the lesion masks provided (which should be
                            stored in directories corresponding to subjects/acquisitions).
                            For now, only 'uniform' is implemented.
                            If 'uniform', the 5th and 95th percentiles of the real distribution of lesion load in
                            subjects/acquisitions are taken, and a uniform distribution is created between these values.
        factor_distribution: Optional: a scipy.stats distribution to sample the intensity factor from.
                            This is not part of the original method, but is used here to 1) ensure we achieve
                            a local hyperintensity, and 2) to allow some more variability in lesion intensity.
                            If None, then this method is not applied.
        im_name: the name of the anat image in the torchio.Subject instance into which the lesion is inserted.
                 Default: 'image'.
        seg_name: the name of the segmentation mask in the torchio.Subject into which the lesion is inserted.
                  Default: 'segmentation'.
        sc_seg_name: the name of the spinal cord segmentation in the torchio.Subject instance. Default: 'sc_seg'.
                    The inserted lesion is cropped to the spinal cord segmentation.
    """
    def __init__(self,
                 lesion_dir: Optional[Path] = None,
                 lesion_paths: Optional[List] = None,
                 load_distribution: Optional[scipy.stats.distributions.rv_frozen] = None,
                 load_distribution_type: Optional[str] = None,
                 factor_distribution: Optional[scipy.stats.rv_continuous] = None,
                 im_name: str = 'image',
                 seg_name: str = 'segmentation',
                 sc_seg_name: str = 'sc_seg',
                 ):
        super().__init__()
        assert (lesion_dir is not None) != (lesion_paths is not None), \
            'One of lesion_dir or lesion_paths must be provided'
        assert (load_distribution is not None) != (load_distribution_type is not None), \
            'One of load_distribution or load_distribution_type must be provided'

        self.im_name = im_name
        self.seg_name = seg_name
        self.sc_seg_name = sc_seg_name
        self.factor_dist = factor_distribution

        # Initialise the lesion paths
        self.lesion_dir = lesion_dir
        if lesion_paths is not None:
            self.lesion_paths = lesion_paths
        else:
            self.lesion_paths = self.init_lesion_paths()

        # Initialise the load distribution
        self.load_distribution_type = load_distribution_type
        if self.load_distribution_type is not None:
            self.load_distribution = self.init_load_distribution()
        else:
            self.load_distribution = load_distribution

    def init_lesion_paths(self) -> List[Tuple[Path, Path]]:
        """ Get a list of all possible lesions and their masks. """
        mask_paths = find_seg_files(self.lesion_dir)

        all_paths = [(p, Path(str(p).replace('_seg.nii.gz', '.nii.gz'))) for p in mask_paths]

        # Check that all paths exist.
        for mask_path, im_path in all_paths:
            if not mask_path.exists() or not im_path.exists():
                msg = f'Could not find lesion image or mask for lesion {im_path}'
                warnings.warn(msg)

        return all_paths

    def init_load_distribution(self) -> Union[scipy.stats.rv_continuous, list]:
        """ Compute the required distribution for the load of the lesion. For now, only uniform is implemented. """
        # To calculate the lesion load in the existing data, we use the lesion masks.
        # The lesion masks should be stored in directories corresponding to subjects/acquisitions.
        volume_ids = [p.parent for p, _ in self.lesion_paths]
        # Create a dictionary to track the volume of each lesion
        lesion_vols = {v: [] for v in set(volume_ids)}

        for mask_path, _ in self.lesion_paths:
            volume_id = mask_path.parent
            mask = tio.LabelMap(mask_path)
            lesion_vols[volume_id].append(mask.data.sum().item())

        # Calculate the lesion load of each volume
        vol_loads = {vol_id: sum(les_vol) for vol_id, les_vol in lesion_vols.items()}

        if self.load_distribution_type == 'uniform':
            # Create a uniform distribution from the lesion loads between 5th and 95th percentiles
            percentiles = np.percentile(list(vol_loads.values()), [5, 95])
            print(f'Initialising uniform load distribution between {percentiles[0]} and {percentiles[1]}')
            load_distribution = scipy.stats.uniform(loc=percentiles[0], scale=percentiles[1] - percentiles[0])
        else:
            raise NotImplementedError('Only uniform load distribution is implemented for now.')

        return load_distribution

    def sample_load_distribution(self) -> float:
        """ Sample a target lesion load from the distribution"""
        # The distribution should either be a scipy distribution or a list of values.
        if isinstance(self.load_distribution, scipy.stats.distributions.rv_frozen):
            return self.load_distribution.rvs()
        elif isinstance(self.load_distribution, list):
            return random.choice(self.load_distribution)
        else:
            raise ValueError('load_distribution must be a scipy distribution or a list of values')

    @staticmethod
    def apply_lesion_augmentation(lesion_subject: tio.Subject) -> tio.Subject:
        """ Apply augmentation directly to the lesion image.
        From paper:
            Flipping - p=0.5 for each dimension
            Rotating - p=0.5 for each dimension, range [1, 89] degrees
            Resizing - range=[0.5, 1.8]  (=> p=1.0 ??)
            Elastic deformation - sigma range [3, 7]
            Brightness - multiply intensity, range [0.9, 1.1]
            Gaussian noise - Addition of N(0, 1) noise
        Args:
            lesion_subject: a subject instance containing a single lesion image and mask.
        Returns:
            The augmented lesion subject.
        """
        augmentation = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), p=0.5),
            tio.RandomAffine(scales=(0.5, 1.8), degrees=(5, 5, 89), p=0.5),
            tio.RandomElasticDeformation(num_control_points=5, max_displacement=4, locked_borders=0, p=0.5),
            tio.RandomGamma(),  # default params - log_gamma = (-0.3, 0.3)
            tio.RandomNoise(mean=0, std=0.1)
        ])
        return augmentation(lesion_subject)

    def get_lesion(self) -> tio.Subject:
        """ Select a random lesion, augment it and return it and its mask."""
        mask_path, im_path = random.choice(self.lesion_paths)
        # Create a Subject instance for the lesion image & mask
        lesion_subject = tio.Subject(lesion_im=tio.ScalarImage(im_path),
                                     lesion_mask=tio.LabelMap(mask_path))

        lesion_subject = self.apply_lesion_augmentation(lesion_subject)

        return lesion_subject

    @staticmethod
    def get_boundary(mask: torch.Tensor) -> torch.Tensor:
        """ Get the boundary of a binary mask by applying an erosion and subtraction."""
        mask = mask > 0
        structuring_element = morph.ball(1)
        eroded = torch.Tensor(morph.binary_erosion(mask[0], footprint=structuring_element)).unsqueeze(0).to(torch.bool)
        return mask * ~eroded

    def local_normalisation(self, lesion_data: torch.Tensor, lesion_mask: torch.Tensor, subject: tio.Subject,
                            z_lower: int, z_upper: int) -> torch.Tensor:
        """ Alter the inserted lesion intensity to create a local hyperintensity based on a multiplicative factor
            drawn from self.factor_distribution. Only applied if self.factor_distribution is not None.
            This is not part of the original LesionMix method. """
        if self.factor_dist is None:
            return lesion_data
        contrast_factor = self.factor_dist.rvs()  # A single value for % contrast between lesion & neighbourhood
        # Extend the bounds to allow for dilation of lesion mask
        extend_lower = 3 if z_lower - 3 > 0 else z_lower
        extend_upper = 3 if z_upper + 3 < subject.spatial_shape[-1] else subject.spatial_shape[-1] - z_upper
        bounds = z_lower - extend_lower, z_upper + extend_upper
        # Start with original lesion mask, extended
        shp = list(lesion_mask.shape)
        shp[-1] = shp[-1] + extend_lower + extend_upper
        neighbourhood_mask = torch.zeros(shp, dtype=torch.uint8)
        neighbourhood_mask[..., extend_lower:extend_lower + lesion_mask.shape[-1]] = lesion_mask
        # Dilate the lesion mask
        structuring_element = morph.ball(3)
        dilated_mask = torch.Tensor(morph.binary_dilation(neighbourhood_mask[0], footprint=structuring_element)).unsqueeze(0)
        neighbourhood_mask = dilated_mask - neighbourhood_mask

        mean_nhood_intensity = (subject[self.im_name].data[..., bounds[0]:bounds[1]][neighbourhood_mask > 0]).mean()
        std_nhood_intensity = (subject[self.im_name].data[..., bounds[0]:bounds[1]][neighbourhood_mask > 0]).std()
        # De-normalise the lesion data to the current neighbourhood distribution
        lesion_data = lesion_data * std_nhood_intensity + mean_nhood_intensity
        # Increase by contrast factor
        return lesion_data * (1 + contrast_factor)

    def add_lesion(self, subject: tio.Subject, lesion: tio.Subject, z_lower: int) -> tio.Subject:
        """ Add the lesion intensity and mask to the subject at the specified z-location, excluding areas
        outside the spinal cord mask.
        Args:
            subject: the subject to add the lesion to, including an intensity image, segmentation mask and SC mask
            lesion: the lesion to add, including an intensity image and segmentation mask (tio.Subject)
            z_lower: the z-location to add the lesion to
        """
        # Get the upper extent of the new lesion location
        z_upper = z_lower + lesion['lesion_im'].shape[-1]
        # Extract the lesion intensity values
        lesion_intensities = lesion['lesion_im'].data
        # Get the relevant area of the spinal cord mask by cropping along z axis
        cropped_sc = subject[self.sc_seg_name].data[..., z_lower:z_upper]
        # Only replace intensities within the SC and lesion mask
        relevant_area = (lesion['lesion_mask'].data * cropped_sc) > 0
        # Get the boundary voxels
        lesion_boundary = self.get_boundary(relevant_area)
        # Replace the relevant part of the subject intensities with the lesion
        mask_excl_boundary = relevant_area * ~lesion_boundary
        # Apply local normalisation, if applicable
        lesion_intensities = self.local_normalisation(lesion_intensities, relevant_area, subject, z_lower, z_upper)
        # Insert the lesion intensity into the relevant part of the subject image
        subject[self.im_name].data[..., z_lower:z_upper][mask_excl_boundary] = (
            lesion_intensities[mask_excl_boundary]
        )
        # Replace the boundary voxels with a weighted average of the lesion and original intensities
        subject[self.im_name].data[..., z_lower:z_upper][lesion_boundary] = (
                0.66 * lesion_intensities[lesion_boundary] +
                0.33 * subject[self.im_name].data[..., z_lower:z_upper][lesion_boundary]
        )
        # Include the lesion mask in the subject segmentation mask
        subject[self.seg_name].data[..., z_lower:z_upper][relevant_area] = 1

        return subject

    def get_load(self, subject: tio.Subject) -> float:
        """ Calculate the current lesion load of the subject. """
        # Get the lesion load of the current subject
        return subject[self.seg_name].data.sum().item()

    def normalise(self, subject: tio.Subject) -> Tuple[tio.Subject, float, float]:
        mu, std = subject[self.im_name].data.float().mean(), subject[self.im_name].data.float().std()
        subject[self.im_name].set_data((subject[self.im_name].data - mu) / std)
        return subject, mu, std

    def denormalise(self, subject: tio.Subject, mean: float, std: float) -> tio.Subject:
        subject[self.im_name].set_data(subject[self.im_name].data.float() * std + mean)
        return subject

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Sample target volume
        target_load = self.sample_load_distribution()
        current_load = self.get_load(subject)
        max_iter = 25  # Set limit mainly to reduce computational time
        n_iter = 0
        while (current_load < target_load) and (n_iter < max_iter):
            # If we are using a contrast factor, then we don't need this normalisation
            if self.factor_dist is None:
                # Apply standardisation (mean=0,std=1) to align the two distributions better
                subject, mu, std = self.normalise(subject)
            lesion = self.get_lesion()
            if lesion['lesion_im'].shape[-1] > subject.spatial_shape[-1]:
                # If the lesion is larger than the subject volume, skip this iteration
                continue
            # Select position for the lesion - this diverges from the paper which uses probability maps
            z = random.randint(0, subject.spatial_shape[-1] - lesion.spatial_shape[-1])
            # Insert lesion data & mask
            subject = self.add_lesion(subject, lesion, z)

            current_load = self.get_load(subject)
            n_iter += 1

        if n_iter == max_iter:
            # If not reaching target load after max_iter, there may be a problem with the data
            warnings.warn(f'Could not reach target load of {target_load} after {max_iter} iterations.')

        if n_iter > 0:
            # Denormalise the subject to return to original intensity distribution
            if self.factor_dist is None:
                subject = self.denormalise(subject, mu, std)

        return subject


class OptionalLesionMixPopulate(LesionMixPopulate):
    @staticmethod
    def check_condition(subject: tio.Subject) -> bool:
        # subject['label'] == 0 => there are no GT lesions in that acquisition
        return subject['label'] == 0

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.check_condition(subject):
            return super().apply_transform(subject)
        return subject


if __name__ == '__main__':
    # Example usage of LesionSCynth
    parser = argparse.ArgumentParser(description='Example usage of LesionSCynth')
    parser.add_argument('--lesion_path', type=Path, required=True,
                        help='Path to a lesion segmentation')
    parser.add_argument('--example_im_path', type=Path, required=True,
                        help='Path to an example image to augment')
    parser.add_argument('--seg_path', type=Path, default=None,
                        help='Path to the corresponding segmentation mask of example_im_path. If None, then blank'
                             'segmentation mask will be created.')
    parser.add_argument('--sc_seg_path', type=Path, default=None,
                        help='Path to the spinal cord segmentation mask of example_im_path. If None, then a '
                             'segmentation mask will be created with all 1.')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Path to directory into which the augmented image and segmentation mask will be saved.'
                             'If None, then nothing will be saved, and the augmented image will be plotted.')
    parser.add_argument('--method', type=str, choices=['LSC', 'LM', 'lesionscynth', 'lesionmix'],
                        default='lesionscynth', help='Method to use for data augmentation.')
    args = parser.parse_args()

    # Fix a seed to ensure reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load an example subject
    example_im = tio.ScalarImage(args.example_im_path)
    example_seg = tio.LabelMap(args.seg_path) if args.seg_path else tio.LabelMap(
        tensor=torch.zeros_like(example_im.data, dtype=torch.uint8),
        affine=example_im.affine
    )
    example_sc_seg = tio.LabelMap(args.sc_seg_path) if args.sc_seg_path else tio.LabelMap(
        tensor=torch.ones_like(example_im.data, dtype=torch.uint8))

    subject = tio.Subject(
        image=example_im,
        segmentation=example_seg,
        sc_seg=example_sc_seg,
        name=str(args.example_im_path).replace('.nii.gz', '')
    )
    a, b = 0.05, 1.0  # Effectively truncated only at left side
    loc = 0.9
    scale = 0.11
    a_transformed = (a - loc) / scale
    b_transformed = (b - loc) / scale
    factor_dist = scipy.stats.truncnorm(a=a_transformed, b=b_transformed, loc=loc, scale=scale)

    synth = LesionSCynth(lesion_paths=[args.lesion_path], modalities=['image'], blur_radius=2, blur_sigma=0.67,
                             gaussian_spatial=True, min_factor_gaussian=0.015, factor_distribution=factor_dist,
                             other_transforms=tio.RandomAffine(scales=0.1, degrees=(5, 5, 45), center='image', p=0.5)
                             )

    # Apply the augmentation
    augmented_subject = synth(subject)

    # Save the augmented image and segmentation mask
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_im_path = os.path.join(args.out_dir, f'{augmented_subject["name"].split("/")[-1]}_aug.nii.gz')
    out_seg_path = os.path.join(args.out_dir, f'{augmented_subject["name"].split("/")[-1]}_aug_seg.nii.gz')
    augmented_subject['image'].save(out_im_path)
    augmented_subject['segmentation'].save(out_seg_path)
    print(f'Saved augmented image to {out_im_path} and segmentation mask to {out_seg_path}')
    
