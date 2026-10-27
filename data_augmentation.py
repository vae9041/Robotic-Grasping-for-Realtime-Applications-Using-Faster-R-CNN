"""Cornell augmentation with one similarity transform for RGB, depth and grasps."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np

from grasp_data import discover_samples, load_grasps, read_split
from grasp_geometry import GraspRectangle as GeometryRectangle


@dataclass
class AugmentationConfig:
    rotation_limit: float = 45
    crop_ratio_min: float = 0.7
    crop_ratio_max: float = 1.0
    zoom_min: float = 0.8
    zoom_max: float = 1.2
    num_augmentations: int = 5
    output_size: tuple = (640, 480)
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2


class GraspRectangle:
    """Compatibility wrapper: augmentation angles are clockwise image degrees."""
    def __init__(self, center_x, center_y, angle, width, height, quality=1.0):
        self.center_x, self.center_y = float(center_x), float(center_y)
        self.angle = float(angle) % 180
        self.width, self.height, self.quality = float(width), float(height), quality

    def get_corners(self):
        return GeometryRectangle(self.center_x, self.center_y, self.width,
                                 self.height, self.angle).to_corners()

    @classmethod
    def from_corners(cls, corners, quality=1.0):
        rectangle = GeometryRectangle.from_corners(corners)
        return cls(rectangle.center_x, rectangle.center_y, math.degrees(rectangle.angle),
                   rectangle.width, rectangle.height, quality)

    def is_valid(self, image_width, image_height, min_size=1):
        corners = self.get_corners()
        return bool(np.isfinite(corners).all() and min(self.width, self.height) >= min_size
                    and (corners >= 0).all()
                    and (corners[:, 0] < image_width).all()
                    and (corners[:, 1] < image_height).all())


class CornellDataset:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.samples = discover_samples([dict(type='original_cornell', path=dataset_path)])
        for sample in self.samples:
            pos = Path(sample['pos_grasps_path'])
            base = pos.name.removesuffix('cpos.txt')
            sample['depth_path'] = next((str(pos.with_name(base + suffix))
                                         for suffix in ('d.png', 'd.tiff', 'd.tif')
                                         if pos.with_name(base + suffix).exists()), None)

    def load_grasps(self, pos_path, neg_path=None):
        # Unsuccessful annotations are deliberately excluded from positive targets.
        grasps = load_grasps(dict(format='cornell', pos_grasps_path=pos_path))
        return [GraspRectangle(g[0], g[1], math.degrees(g[4]), g[2], g[3]) for g in grasps]


class JacquardDataset:
    def __init__(self, dataset_path):
        self.samples = discover_samples([dict(type='original_jacquard', path=dataset_path)])

    def load_grasps(self, grasps_path):
        grasps = load_grasps(dict(format='jacquard', grasps_path=grasps_path))
        return [GraspRectangle(g[0], g[1], math.degrees(g[4]), g[2], g[3]) for g in grasps]


class GraspAugmenter:
    def __init__(self, config):
        self.config = config
        if not (0 < config.crop_ratio_min <= config.crop_ratio_max <= 1):
            raise ValueError('Crop ratios must satisfy 0 < min <= max <= 1')
        if not (0 < config.zoom_min <= config.zoom_max):
            raise ValueError('Zoom factors must be positive and ordered')

    def augment_sample(self, rgb_image, depth_image, grasps):
        if depth_image.shape[:2] != rgb_image.shape[:2]:
            raise ValueError('RGB and depth must be registered with equal image dimensions')
        height, width = rgb_image.shape[:2]
        out_w, out_h = self.config.output_size
        crop = random.uniform(self.config.crop_ratio_min, self.config.crop_ratio_max)
        zoom = random.uniform(self.config.zoom_min, self.config.zoom_max)
        scale = min(out_w / width, out_h / height) * zoom / crop
        angle = random.uniform(-self.config.rotation_limit, self.config.rotation_limit)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        jitter_x = random.uniform(-1, 1) * out_w * (1 - crop) / 2
        jitter_y = random.uniform(-1, 1) * out_h * (1 - crop) / 2
        matrix[:, 2] += [(out_w - width) / 2 + jitter_x, (out_h - height) / 2 + jitter_y]
        rgb = cv2.warpAffine(rgb_image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
        depth = cv2.warpAffine(depth_image, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST)
        transformed = []
        for grasp in grasps:
            if grasp.quality <= .5:
                continue
            corners = np.column_stack((grasp.get_corners(), np.ones(4))) @ matrix.T
            new_grasp = GraspRectangle.from_corners(corners, grasp.quality)
            if new_grasp.is_valid(out_w, out_h):
                transformed.append(new_grasp)
        # Photometric changes never affect depth or geometric annotations.
        contrast = random.uniform(1 - self.config.contrast_limit, 1 + self.config.contrast_limit)
        brightness = random.uniform(-self.config.brightness_limit, self.config.brightness_limit) * 255
        rgb = np.clip(rgb.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        return rgb, depth, transformed

    def _opencv_augment(self, image, grasps):
        rgb, _, transformed = self.augment_sample(image, np.zeros(image.shape[:2], np.uint16), grasps)
        return rgb, transformed


def save_augmented_data(output_dir, sample_id, aug_id, rgb_image, depth_image, grasps,
                        dataset_type, source_id=None):
    if not source_id:
        raise ValueError('source_id is required to keep augmentations in their original training group')
    output_dir = Path(output_dir)
    for folder in ('rgb', 'depth', 'annotations'):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    base = f'{sample_id}_aug_{aug_id}'
    if not cv2.imwrite(str(output_dir / 'rgb' / (base + '.png')), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)):
        raise OSError('Could not write augmented RGB image')
    if not cv2.imwrite(str(output_dir / 'depth' / (base + '.tiff')), depth_image):
        raise OSError('Could not write augmented depth image')
    annotations = dict(image_id=base, dataset_source=dataset_type, source_id=source_id,
                       geometry_version=2, grasps=[])
    for grasp in grasps:
        if grasp.quality <= .5:
            continue
        annotations['grasps'].append(dict(center_x=grasp.center_x, center_y=grasp.center_y,
                                          angle=grasp.angle, width=grasp.width, height=grasp.height,
                                          quality=grasp.quality, corners=grasp.get_corners().tolist()))
    (output_dir / 'annotations' / (base + '.json')).write_text(json.dumps(annotations, indent=2) + '\n')


def augment_datasets(cornell_path, jacquard_path, output_path, config, split_manifest=None):
    dataset = CornellDataset(cornell_path)
    training_ids = set(read_split(split_manifest)['train_ids']) if split_manifest else None
    augmenter = GraspAugmenter(config)
    generated = 0
    for sample in dataset.samples:
        if training_ids is not None and sample['sample_id'] not in training_ids:
            continue
        image = cv2.imread(sample['rgb_path'])
        if image is None:
            raise ValueError(f"Cannot read {sample['rgb_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = (cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
                 if sample['depth_path'] else np.zeros(image.shape[:2], np.uint16))
        if depth is None:
            raise ValueError(f"Cannot read {sample['depth_path']}")
        grasps = dataset.load_grasps(sample['pos_grasps_path'])
        # The source path gives stable names independent of file enumeration order.
        name = sample['sample_id'].replace('/', '_')
        for aug_id in range(config.num_augmentations):
            rgb, aug_depth, augmented = augmenter.augment_sample(image, depth, grasps)
            if augmented:
                save_augmented_data(output_path, name, aug_id, rgb, aug_depth, augmented,
                                    'cornell', source_id=sample['sample_id'])
                generated += 1
    print(f'Generated {generated} Cornell augmentations in {output_path}')


def visualize_grasps(image, grasps, output_path):
    result = image.copy()
    for grasp in grasps:
        cv2.polylines(result, [grasp.get_corners().astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description='Augment Cornell with consistent oriented geometry')
    parser.add_argument('--cornell_path', default='./cornell_dataset')
    parser.add_argument('--jacquard_path', default='./jacquard_dataset', help='Unused; Cornell-only augmentation')
    parser.add_argument('--output_path', default='./augmented_dataset')
    parser.add_argument('--split_manifest', help='Generate training images only from this saved split')
    parser.add_argument('--num_augmentations', type=int, default=5)
    parser.add_argument('--rotation_limit', type=float, default=45)
    parser.add_argument('--crop_ratio_min', type=float, default=.7)
    parser.add_argument('--crop_ratio_max', type=float, default=1.)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    config = AugmentationConfig(num_augmentations=args.num_augmentations, rotation_limit=args.rotation_limit,
                                crop_ratio_min=args.crop_ratio_min, crop_ratio_max=args.crop_ratio_max)
    augment_datasets(args.cornell_path, args.jacquard_path, args.output_path, config, args.split_manifest)
    if args.visualize:
        root = Path(args.output_path)
        for annotation in sorted((root / 'annotations').glob('*.json'))[:5]:
            metadata = json.loads(annotation.read_text())
            image = cv2.cvtColor(cv2.imread(str(root / 'rgb' / (annotation.stem + '.png'))), cv2.COLOR_BGR2RGB)
            grasps = [GraspRectangle.from_corners(g['corners']) for g in metadata['grasps']]
            visualize_grasps(image, grasps, root / (annotation.stem + '_preview.png'))


if __name__ == '__main__':
    main()
