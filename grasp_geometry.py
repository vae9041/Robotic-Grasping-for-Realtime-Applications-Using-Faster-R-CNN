"""Grasp geometry in image coordinates: x right, y down, angles clockwise.

Internal grasps are (cx, cy, opening, jaw_size, theta_radians), with theta
periodic modulo pi. Width always follows corner 0 -> corner 1; it is never
swapped with height based on which edge is longer.
"""
import math

import cv2
import numpy as np


class GraspRectangle:
    def __init__(self, center_x, center_y, width, height, angle=0.0, format_type='jacquard'):
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.width = float(width)
        self.height = float(height)
        self.angle = (math.radians(angle) if format_type == 'jacquard' else float(angle)) % math.pi

    @classmethod
    def from_array(cls, grasp):
        return cls(*map(float, grasp), format_type='radians')

    def as_array(self):
        return np.array([self.center_x, self.center_y, self.width, self.height, self.angle], dtype=np.float32)

    @classmethod
    def from_bbox(cls, x_min, y_min, x_max, y_max, angle=0.0):
        """Build a rectangle with these side lengths; an AABB is NOT an oriented grasp."""
        return cls((x_min + x_max) / 2, (y_min + y_max) / 2,
                   x_max - x_min, y_max - y_min, angle, 'radians')

    @classmethod
    def from_corners(cls, corners, y_down=True):
        corners = np.asarray(corners, dtype=np.float64)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError('Expected four finite, cyclically ordered grasp corners')
        edge = corners[1] - corners[0]
        theta = math.atan2(edge[1] if y_down else -edge[1], edge[0])
        width = float(np.linalg.norm(edge))
        height = float(np.linalg.norm(corners[2] - corners[1]))
        if min(width, height) <= 0:
            raise ValueError('Grasp dimensions must be positive')
        return cls(*corners.mean(axis=0), width, height, theta, 'radians')

    def to_corners(self):
        w, h = self.width / 2, self.height / 2
        local = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
        c, s = math.cos(self.angle), math.sin(self.angle)
        return local @ np.array([[c, s], [-s, c]]) + [self.center_x, self.center_y]

    def to_bbox(self):
        corners = self.to_corners()
        return (*corners.min(axis=0), *corners.max(axis=0))

    def angle_error(self, other):
        delta = (self.angle - other.angle + math.pi / 2) % math.pi - math.pi / 2
        return abs(math.degrees(delta))

    def intersection_over_union(self, other):
        """True polygon IoU. Never silently substitute axis-aligned overlap."""
        a, b = self.to_corners().astype(np.float32), other.to_corners().astype(np.float32)
        intersection = max(0.0, float(cv2.intersectConvexConvex(a, b)[0]))
        union = self.width * self.height + other.width * other.height - intersection
        return float(np.clip(intersection / union, 0, 1)) if union > 0 else 0.0


def aabb_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    union = float(np.prod(np.maximum(0, a[2:] - a[:2])) +
                  np.prod(np.maximum(0, b[2:] - b[:2]))) - intersection
    return intersection / union if union > 0 else 0.0


def letterbox(image, target_size=(640, 480)):
    """Resize with one exact scale and pad; return image, scale and x/y offset."""
    height, width = image.shape[:2]
    out_width, out_height = target_size
    scale = min(out_width / width, out_height / height)
    offset = np.array([(out_width - scale * width) / 2,
                       (out_height - scale * height) / 2])
    matrix = np.array([[scale, 0, offset[0]], [0, scale, offset[1]]], dtype=np.float64)
    return cv2.warpAffine(image, matrix, (out_width, out_height)), scale, offset


def transform_grasps(grasps, scale, offset):
    result = np.asarray(grasps, dtype=np.float32).reshape(-1, 5).copy()
    result[:, :2] = result[:, :2] * scale + offset
    result[:, 2:4] *= scale
    return result


def grasp_match(prediction, ground_truth, iou_threshold=0.25, angle_threshold=30.0):
    """Match ANY one GT satisfying both conditions, not just the largest-IoU GT.

    Returns (matched_index, IoU, angular_error_degrees). Among valid matches,
    select the largest IoU for deterministic one-to-one per-grasp evaluation.
    """
    pred = GraspRectangle.from_array(prediction)
    best = (-1, 0.0, 90.0)
    for index, grasp in enumerate(ground_truth):
        gt = GraspRectangle.from_array(grasp)
        error = pred.angle_error(gt)
        if error > angle_threshold + 1e-5:
            continue
        iou = pred.intersection_over_union(gt)
        if iou > iou_threshold and iou > best[1]:
            best = (index, iou, error)
    return best
