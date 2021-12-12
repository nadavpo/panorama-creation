"""Projective Homography and Panorama Solution."""
from math import nan
import random
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple

from numpy.linalg import svd
from scipy.interpolate import griddata
import cv2
import matplotlib.pyplot as plt

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""

    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        n = match_p_dst.shape[1]
        A2 = np.zeros((2 * match_p_dst.shape[1], 9))
        for i in range(0, match_p_dst.shape[1]):
            vector = np.concatenate((match_p_src[:, i], [1]))
            A2[2 * i, :] = np.concatenate((vector, np.zeros(3), -1 * match_p_dst[0, i] * vector))
            A2[2 * i + 1, :] = np.concatenate((np.zeros(3), vector, -1 * match_p_dst[1, i] * vector))

        U, S, Vh = np.linalg.svd(A2)
        min_eigen = Vh[-1, :]
        homography = np.reshape(min_eigen, (3, 3))
        
        return homography

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # get the elements in the transform matrix
        H = homography
        h0 = H[0, 0]
        h1 = H[0, 1]
        h2 = H[0, 2]
        h3 = H[1, 0]
        h4 = H[1, 1]
        h5 = H[1, 2]
        h6 = H[2, 0]
        h7 = H[2, 1]
        h8 = H[2, 2]

        new_image = np.zeros(dst_image_shape)
        for row in range(0, src_image.shape[0]):
            for col in range(0, src_image.shape[1]):
                tx = (h0 * col + h1 * row + h2)
                ty = (h3 * col + h4 * row + h5)
                tz = (h6 * col + h7 * row + h8)
                px = int(tx / tz)
                py = int(ty / tz)
                Z = int(1 / tz)
                if px >= dst_image_shape[0] or py >= dst_image_shape[1] or px <= 0 or py <= 0:
                    continue
                new_image[py, px, :] = src_image[row, col, :] / 255.0

        return new_image

    @staticmethod
    def compute_forward_homography_pnts(
            homography: np.ndarray,
            src_points: np.ndarray
    ) -> np.ndarray:
        H = homography
        h0 = H[0, 0]
        h1 = H[0, 1]
        h2 = H[0, 2]
        h3 = H[1, 0]
        h4 = H[1, 1]
        h5 = H[1, 2]
        h6 = H[2, 0]
        h7 = H[2, 1]
        h8 = H[2, 2]
        out_pts = []
        for pnt in src_points:
            tx = (h0 * pnt[0] + h1 * pnt[1] + h2)
            ty = (h3 * pnt[0] + h4 * pnt[1] + h5)
            tz = (h6 * pnt[0] + h7 * pnt[1] + h8)
            px = int(tx / tz)
            py = int(ty / tz)
            out_pts.append([px, py])

        return np.asarray(out_pts)

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        Create a meshgrid of columns and rows.
        Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        Convert the coordinates into integer values and clip them
        according to the destination image size.
        Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        new_image = np.zeros(dst_image_shape)
        y = np.linspace(0, src_image.shape[0] - 1, src_image.shape[0])
        x = np.linspace(0, src_image.shape[1] - 1, src_image.shape[1])
        yy, xx = np.meshgrid(y, x, sparse=False, indexing='ij')
        pixel_loc = np.zeros([src_image.shape[0], src_image.shape[1], 3])
        pixel_loc[:, :, 0] = xx
        pixel_loc[:, :, 1] = yy
        pixel_loc[:, :, 2] = 1
        pixel_loc = pixel_loc.astype(int)
        im_trans = np.einsum("ijk,ak->ija", pixel_loc, homography)
        im_trans[:, :, 0] = im_trans[:, :, 0] / im_trans[:, :, 2]
        im_trans[:, :, 1] = im_trans[:, :, 1] / im_trans[:, :, 2]
        im_trans = np.round(im_trans[:, :, :2]).astype(int)

        pixel_locX = pixel_loc[:, :, 0].flatten()
        pixel_locy = pixel_loc[:, :, 1].flatten()

        im_transX = im_trans[:, :, 0].flatten()
        im_transy = im_trans[:, :, 1].flatten()

        valid_ind = (im_transX > 0) & (im_transX < dst_image_shape[1]) & \
                    (im_transy > 0) & (im_transy < dst_image_shape[0])

        pixel_locX = pixel_locX[valid_ind]
        pixel_locy = pixel_locy[valid_ind]
        im_transX = im_transX[valid_ind]
        im_transy = im_transy[valid_ind]

        new_image[im_transy, im_transX, :] = src_image[pixel_locy, pixel_locX, :] / 255.0

        return new_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        u = (homography[0, 0] * match_p_src[0] + homography[0, 1] * match_p_src[1] + homography[0, 2]) / \
            (homography[2, 0] * match_p_src[0] + homography[2, 1] * match_p_src[1] + homography[2, 2])
        v = (homography[1, 0] * match_p_src[0] + homography[1, 1] * match_p_src[1] + homography[1, 2]) / \
            (homography[2, 0] * match_p_src[0] + homography[2, 1] * match_p_src[1] + homography[2, 2])
        dist = pow(pow(match_p_dst[0, :] - u, 2) + pow(match_p_dst[1, :] - v, 2), 0.5)
        fit_percent = len(dist[dist < max_err]) / match_p_dst.shape[1]

        if fit_percent > 0:
            dist_mse = np.average(dist[dist < max_err])
        else:
            dist_mse = 10 ** 9
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        H = homography
        h0 = H[0, 0]
        h1 = H[0, 1]
        h2 = H[0, 2]
        h3 = H[1, 0]
        h4 = H[1, 1]
        h5 = H[1, 2]
        h6 = H[2, 0]
        h7 = H[2, 1]
        h8 = H[2, 2]
        dist = []
        for i in range(0, match_p_dst.shape[1]):
            tx = (h0 * match_p_src[0, i] + h1 * match_p_src[1, i] + h2)
            ty = (h3 * match_p_src[0, i] + h4 * match_p_src[1, i] + h5)
            tz = (h6 * match_p_src[0, i] + h7 * match_p_src[1, i] + h8)
            px = int(tx / tz)
            py = int(ty / tz)
            Z = int(1 / tz)
            dist.append(
                pow(pow(match_p_dst[0, i] - px, 2) + pow(match_p_dst[1, i] - py, 2), 0.5))
        dist = np.asarray(dist)
        # mp_src_meets_model = np.asarray(match_p_src[dist < max_err,:])
        mp_src_meets_model = match_p_src[:, dist < max_err]
        mp_dst_meets_model = match_p_dst[:, dist < max_err]
        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        random.seed(6)
        homography = []
        # # use class notations:
        w = inliers_percent
        t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        p = 0.99
        # # the minimal probability of points which meets with the model
        d = 0.5
        # # number of points sufficient to compute the model
        n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        is_valid = False
        n_orig_match_points = match_p_src.shape[1]
        best_err = 10 ** 9
        for _ in range(1, k + 1):
            inds = sample(range(1, n_orig_match_points), n)
            curr_p_src = match_p_src[:, inds]
            curr_p_dst = match_p_dst[:, inds]
            naive_homography = self.compute_homography_naive(curr_p_src, curr_p_dst)
            fit_percent, dist_mse = self.test_homography(naive_homography, match_p_src, match_p_dst, t)
            if bool(fit_percent > d) & bool(dist_mse < best_err):
                mp_src_meets_model, mp_dst_meets_model = self.meet_the_model_points(naive_homography, match_p_src,
                                                                                    match_p_dst, max_err)
                homography = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
                best_err = dist_mse
                is_valid = True
        #homography/=homography[2,2]
        if not is_valid:
            raise Exception('faild to find homography in ransac')
        
        return homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        Create a mesh-grid of columns and rows of the destination image.
        Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        Compute the corresponding coordinates in the source image using
        the backward projective homography.
        Create the mesh-grid of source image coordinates.
        For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        new_image = np.zeros(dst_image_shape)
        y = np.linspace(0, dst_image_shape[0] - 1, dst_image_shape[0])
        x = np.linspace(0, dst_image_shape[1] - 1, dst_image_shape[1])
        yy, xx = np.meshgrid(y, x, sparse=False, indexing='ij')
        dst_pixel_loc = np.zeros([dst_image_shape[0], dst_image_shape[1], 3])
        dst_pixel_loc[:, :, 0] = xx
        dst_pixel_loc[:, :, 1] = yy
        dst_pixel_loc[:, :, 2] = 1
        dst_pixel_loc = dst_pixel_loc.astype(int)
        im_trans = np.einsum("ijk,ak->ija", dst_pixel_loc, backward_projective_homography)
        im_trans[:, :, 0] = im_trans[:, :, 0] / im_trans[:, :, 2]
        im_trans[:, :, 1] = im_trans[:, :, 1] / im_trans[:, :, 2]
        im_trans = im_trans[:, :, :2]

        # src
        y = np.linspace(0, src_image.shape[0] - 1, src_image.shape[0])
        x = np.linspace(0, src_image.shape[1] - 1, src_image.shape[1])
        yy, xx = np.meshgrid(y, x, sparse=False, indexing='ij')
        src_pixel_loc = np.zeros([src_image.shape[0], src_image.shape[1], 2])
        src_pixel_loc[:, :, 0] = xx
        src_pixel_loc[:, :, 1] = yy
        src_pixel_loc = src_pixel_loc.astype(int)

        src_pixel_locx = src_pixel_loc[:, :, 0].flatten()
        src_pixel_locy = src_pixel_loc[:, :, 1].flatten()
        tot_src_pixel_loc = np.zeros((src_pixel_locy.shape[0], 2))
        tot_src_pixel_loc[:, 1] = src_pixel_locx
        tot_src_pixel_loc[:, 0] = src_pixel_locy

        im_transX = im_trans[:, :, 0].flatten()
        im_transy = im_trans[:, :, 1].flatten()

        valid_ind = (im_transX > 0) & (im_transX < src_image.shape[1]) & \
                    (im_transy > 0) & (im_transy < src_image.shape[0])

        im_transX = im_transX[valid_ind]
        im_transy = im_transy[valid_ind]
        new_image[:, :, 0] = griddata(tot_src_pixel_loc, src_image[:, :, 0].flatten(),
                                      (im_trans[:, :, 1], im_trans[:, :, 0]), \
                                      method='cubic', fill_value=nan)
        new_image[:, :, 1] = griddata(tot_src_pixel_loc, src_image[:, :, 1].flatten(),
                                      (im_trans[:, :, 1], im_trans[:, :, 0]), \
                                      method='cubic', fill_value=nan)
        new_image[:, :, 2] = griddata(tot_src_pixel_loc, src_image[:, :, 2].flatten(),
                                      (im_trans[:, :, 1], im_trans[:, :, 0]), \
                                      method='cubic', fill_value=nan)

        return new_image

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([0, 0, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 0, 1])
        src_edges['lower left corner'] = np.array([0, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = np.matmul(homography, corner_location)
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        Build the translation matrix from the pads.
        Compose the backward homography and the translation matrix together.
        Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # (1) Build the translation matrix from the pads.
        adding_h = np.eye(3)
        adding_h[0, 2] =  -1 * pad_left
        adding_h[1, 2] =  -1 * pad_up
        # (2) Compose the backward homography and the translation matrix together.
        final_homography = np.matmul(backward_homography, adding_h)
        # (3) Scale the homography as learnt in class.
        final_homography /= np.linalg.norm(final_homography)
        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        Compute the forward homography and the panorama shape.
        Compute the backward homography.
        Add the appropriate translation to the homography so that the
        source image will plant in place.
        Compute the backward warping with the appropriate translation.
        Create the an empty panorama image and plant there the
        destination image.
        place the backward warped image in the indices where the panorama
        image is zero.


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # Compute the forward homography and the panorama shape.
        homof = Solution.compute_homography(self, match_p_src, match_p_dst, inliers_percent, max_err)
        panorama_shape = Solution.find_panorama_shape(src_image, dst_image, homography=homof)
        # Compute the backward homography.
        homob = Solution.compute_homography(self, match_p_dst, match_p_src, inliers_percent, max_err)
        # Add the appropriate translation to the homography so that the source image will plant in place.
        homobt = Solution.add_translation_to_backward_homography(homob, panorama_shape[2].pad_left,
                                                                 panorama_shape[2].pad_up)
        # Compute the backward warping with the appropriate translation.
        panorama_shape1 = (panorama_shape[0], panorama_shape[1], 3)
        bacwardwarped = Solution.compute_backward_mapping(homobt, src_image, panorama_shape1)
        # Create the an empty panorama image and plant there the destination image.
        panoram = np.zeros(panorama_shape1)
        # insert the dest image in the panoram.
        start_y_pix = panorama_shape[2].pad_up
        start_x_pix = panorama_shape[2].pad_left
        panoram[start_y_pix:start_y_pix + dst_image.shape[0], start_x_pix:start_x_pix + dst_image.shape[1],
        :] = dst_image
        # place the backward warped image in the indices where the panorama image is zero.
        for col in range(0, panoram.shape[0]):
            for row in range(0, panoram.shape[1]):
                if (any(np.isnan(panoram[col, row, :])) or any(panoram[col, row, :] == 0)) and not (
                any(np.isnan(bacwardwarped[col, row, :]))):
                    panoram[col, row, :] = bacwardwarped[col, row, :]
        panoram = np.clip(panoram, 0, 255)
        panoram = panoram / 255.0
        return panoram
