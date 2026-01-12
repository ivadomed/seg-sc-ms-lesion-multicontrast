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