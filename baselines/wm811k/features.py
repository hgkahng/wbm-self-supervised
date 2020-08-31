# -*- coding: utf-8 -*-

"""Hand-crafted features for WM811K."""

import math
import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import hough_line
from skimage.transform import radon
from scipy.signal import resample


class WBMFeatures(object):
    """Base class for WBM feature classes."""
    @staticmethod
    def read_wbm(path: str):
        label = pathlib.Path(path).parent.name        # string
        wbm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H, W)
        return wbm, label

    @staticmethod
    def only_valids(wbm: np.ndarray):
        return cv2.threshold(wbm, 1, 255, cv2.THRESH_BINARY)[1]

    @staticmethod
    def only_defects(wbm: np.ndarray):
        return cv2.threshold(wbm, 127, 255, cv2.THRESH_BINARY)[1]


class RadonFeatures(WBMFeatures):
    def __init__(self, wbm_or_path: np.ndarray or str, label: str = None):
        """Add function docstring."""
        if isinstance(wbm_or_path, str):
            self.wbm, self.label = self.read_wbm(wbm_or_path)
        elif isinstance(wbm_or_path, np.ndarray):
            assert wbm_or_path.dtype == np.uint8
            self.wbm = wbm_or_path
            self.label = label
        else:
            raise NotImplementedError

        G = radon(self.wbm, theta=np.arange(180), circle=True)
        G = (G - G.min() / G.max() - G.min())

        means = np.mean(G, axis=1)
        means = resample(means, 20)  # uses Fourier transfrom

        stds = np.std(G, axis=1)
        stds = resample(stds, 20)    # uses Fourier transfrom

        self._features = dict()
        self._features.update({f'radon_mean_{i}': v for i, v in enumerate(means)})
        self._features.update({f'radon_std_{i}': v for i, v in enumerate(stds)})

        self.G = G

    @property
    def data(self):
        return self._features


class GeometryFeatures(WBMFeatures):
    def __init__(self, wbm_or_path: np.ndarray or str, label: str = None):
        """Add function docstring."""
        if isinstance(wbm_or_path, str):
            self.wbm, self.label = self.read_wbm(wbm_or_path)
        elif isinstance(wbm_or_path, np.ndarray):
            assert wbm_or_path.dtype == np.uint8
            self.wbm = wbm_or_path
            self.label = label
        else:
            raise NotImplementedError

        sa_roi, sa_mask, sa_area = self.find_cc_max_area(self.wbm)
        sp_roi, sp_mask, sp_peri = self.find_cc_max_perimeter(self.wbm)

        self._features = dict()
        self._features['area_ratio'] = sa_area / self.wbm_area
        self._features['perimeter_ratio'] = sp_peri / self.wbm_radius
        self._features['sa_distance'] = sum([(a - b)**2 for a, b in zip(self.find_center(sa_mask), self.wbm_center)])
        self._features['sp_distance'] = sum([(a - b)**2 for a, b in zip(self.find_center(sp_mask), self.wbm_center)])
        self._features['sa_major_axis_ratio'] = self.ellipse_axis_lengths(sa_roi, mode='major') / self.wbm_radius
        self._features['sa_minor_axis_ratio'] = self.ellipse_axis_lengths(sa_roi, mode='minor') / self.wbm_radius
        self._features['sp_major_axis_ratio'] = self.ellipse_axis_lengths(sp_roi, mode='major') / self.wbm_radius
        self._features['sp_minor_axis_ratio'] = self.ellipse_axis_lengths(sp_roi, mode='minor') / self.wbm_radius
        self._features['sa_solidity'] = self.solidity(sa_roi)
        self._features['sp_solidity'] = self.solidity(sp_roi)
        self._features['sa_eccentricity'] = self.eccentricity(sa_roi)
        self._features['sp_eccentricity'] = self.eccentricity(sp_roi)
        self._features['defect_ratio'] = (self.wbm == 255).sum() / (self.wbm > 0).sum()
        self._features['hough_max'] = hough_line(self.wbm)[0].max()

        self.sa_roi = sa_roi
        self.sp_roi = sp_roi
        self.sa_mask = sa_mask
        self.sp_mask = sp_mask

    @classmethod
    def find_cc_max_area(cls, wbm: np.ndarray):
        """Find the connected component with the largest area."""
        wbm = cls.only_defects(wbm)
        _, labels, stats, _ = \
            cv2.connectedComponentsWithStats(wbm, 4, cv2.CV_32S)

        i = stats[1:, cv2.CC_STAT_AREA].argmax() + 1  # ignore background
        roi = cls.get_roi(wbm, stats, i)
        mask = np.zeros_like(wbm)
        mask[labels != i] = 0
        mask[labels == i] = 1
        area = stats[i, cv2.CC_STAT_AREA]

        return roi, mask, area  # np.uint8, np.int32, int

    @classmethod
    def find_cc_max_perimeter(cls, wbm: np.ndarray):
        """Find the connected component with the largest arc length."""
        wbm = cls.only_defects(wbm)
        num_labels, labels, stats, _ = \
            cv2.connectedComponentsWithStats(wbm, 4, cv2.CV_32S)

        max_peri = 0.0
        max_roi  = None
        max_mask = None

        for i in range(num_labels):

            if i == 0:
                continue  # ignore background

            roi = cls.get_roi(wbm, stats, i)
            contour, _ = cls.find_largest_contour(roi)
            perimeter = cv2.arcLength(contour, True)

            if max_roi is None:
                max_roi = roi

            if max_mask is None:
                max_mask = np.zeros_like(wbm)
                max_mask[labels != i] = 0
                max_mask[labels == i] = 1

            if perimeter > max_peri:
                max_peri = perimeter
                max_roi = roi
                max_mask = np.zeros_like(wbm)
                max_mask[labels != i] = 0
                max_mask[labels == i] = 1

        return max_roi, max_mask, max_peri  # np.uint8, np.int32, float

    @classmethod
    def find_center(cls, mask: np.ndarray):
        contour, _ = cls.find_largest_contour(mask)
        (x, y), _ = cv2.minEnclosingCircle(contour)
        return x, y

    @classmethod
    def ellipse_axis_lengths(cls, roi: np.ndarray, mode: str = 'major'):
        contour, _ = cls.find_largest_contour(roi)
        try:
            (x, y), (w, h), angle = cv2.fitEllipse(contour)  # pylint: disable=unused-variable
            if mode == 'major':
                return max([w, h])
            elif mode == 'minor':
                return min([w, h])
        except: # pylint: disable=bare-except
            return np.nan

    @classmethod
    def solidity(cls, roi: np.ndarray):
        contour, area = cls.find_largest_contour(roi)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        try:
            return float(area) / float(hull_area)
        except ZeroDivisionError:
            return np.nan

    @classmethod
    def eccentricity(cls, roi: np.ndarray):
        contour, _ = cls.find_largest_contour(roi)
        try:
            (x, y), (w, h), angle = cv2.fitEllipse(contour)  # pylint: disable=unused-variable
            major = max([w, h])
            minor = min([w, h])
            return math.sqrt(1 - minor**2/major**2)
        except:  # pylint: disable=bare-except
            return np.nan


    @staticmethod
    def get_roi(wbm: np.ndarray, stats: np.ndarray, i: int):
        assert stats.shape[1] == 5
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w  = stats[i, cv2.CC_STAT_WIDTH]
        return wbm[y:y+h, x:x+w]

    @staticmethod
    def find_largest_contour(roi: np.ndarray):
        contours, _ = cv2.findContours(roi, 1, 2)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        i, v = max(enumerate(areas), key=lambda k: k[1])
        return contours[i], v  # contour, area

    @property
    def wbm_area(self):
        wbm = self.only_valids(self.wbm)
        _, area = self.find_largest_contour(wbm)
        return area

    @property
    def wbm_radius(self):
        wbm = self.only_valids(self.wbm)
        contour, _ = self.find_largest_contour(wbm)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return radius

    @property
    def wbm_center(self):
        return [s // 2 for s in self.wbm.shape]

    @property
    def data(self):
        return self._features

    @classmethod
    def visualize(cls, wbm_or_path: np.ndarray or str, cmap: object = plt.cm.binary):
        geo_feat = cls(wbm_or_path)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        ax1.imshow(geo_feat.wbm, cmap=cmap)
        ax2.imshow(cls.only_defects(geo_feat.wbm), cmap=cmap)
        ax3.imshow(geo_feat.sa_mask, cmap=cmap)
        ax4.imshow(geo_feat.sp_mask, cmap=cmap)
        ax1.set_title(f'Label: {geo_feat.label}')
        ax2.set_title('Only defects')
        ax3.set_title('ROI with maximal area')
        ax4.set_title('ROI with maximal perimeter')
        plt.show(fig)


class DensityFeatures(WBMFeatures):
    def __init__(self, wbm_or_path: np.ndarray or str, label: str = None):
        """Add function docstring."""
        if isinstance(wbm_or_path, str):
            self.wbm, self.label = self.read_wbm(wbm_or_path)
        elif isinstance(wbm_or_path, np.ndarray):
            assert wbm_or_path.dtype == np.uint8
            self.wbm = wbm_or_path
            self.label = label
        else:
            raise NotImplementedError

        h, w = self.wbm.shape
        hs = np.linspace(0, h, 7, endpoint=True).astype(int)
        ws = np.linspace(0, w, 7, endpoint=True).astype(int)

        self._features = {}
        self._features['d_top'] = (self.wbm[hs[0]:hs[1], :] == 255).sum() / (self.wbm[hs[0]:hs[1], :] > 0).sum()
        self._features['d_bottom'] = (self.wbm[hs[-2]:hs[-1], :] == 255).sum() / (self.wbm[hs[-2]:hs[-1], :] > 0).sum()
        self._features['d_left'] = (self.wbm[:, ws[0]:ws[1]] == 255).sum() / (self.wbm[:, ws[0]:ws[1]] > 0).sum()
        self._features['d_right'] = (self.wbm[:, ws[-2]:ws[-1]] == 255).sum() / (self.wbm[:, ws[-2]:ws[-1]] > 0).sum()

        for i in (1, 2, 3, 4):
            for j in (1, 2, 3, 4):
                region = self.wbm[hs[i]:hs[i+1], ws[j]:ws[j+1]]
                self._features[f'd_{i}{j}'] = (region == 255).sum() / (region > 0).sum()

    @property
    def data(self):
        return self._features
