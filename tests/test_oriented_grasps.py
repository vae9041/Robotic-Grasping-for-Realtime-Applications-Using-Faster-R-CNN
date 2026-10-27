"""Regression tests for real grasp geometry, annotation provenance and model gradients.

Run: python -m unittest discover -s tests -v
No datasets, pretrained downloads, camera or CUDA are required.
"""
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch
from torch import nn

from data_augmentation import (AugmentationConfig, GraspAugmenter,
                               GraspRectangle as AugRectangle, save_augmented_data)
from grasp_data import create_split, discover_samples, load_grasps, select_indices, validate_split
from grasp_geometry import GraspRectangle, aabb_iou, letterbox
from grasp_metrics import score_image
from oriented_model import (GRASP_FORMAT, GraspRCNNTransform, OrientedFastRCNNPredictor,
                            decode_grasps, encode_grasps, load_training_weights)
from train import GraspDataset, build_training_split, checkpoint_info, evaluate


def rectangle(angle=30, center=(64, 48), size=(40, 12)):
    return GraspRectangle(*center, *size, angle)


def write_cornell(root, number, grasp=None, count=1):
    folder = root / '01'
    folder.mkdir(parents=True, exist_ok=True)
    base = f'pcd{number:04d}'
    cv2.imwrite(str(folder / (base + 'r.png')), np.zeros((96, 128, 3), np.uint8))
    grasp = grasp or rectangle()
    text = '\n'.join(f'{x} {y}' for x, y in grasp.to_corners()) + '\n'
    path = folder / (base + 'cpos.txt')
    path.write_text(text * count)
    # A distinct unsuccessful rectangle must not become a positive annotation.
    (folder / (base + 'cneg.txt')).write_text('1 1\n2 1\n2 2\n1 2\n')
    return path


def write_jacquard(root, object_id, view):
    folder = root / 'part0' / object_id
    folder.mkdir(parents=True, exist_ok=True)
    base = f'{view}_{object_id}'
    cv2.imwrite(str(folder / (base + '_RGB.png')), np.zeros((100, 100, 3), np.uint8))
    (folder / (base + '_grasps.txt')).write_text('50;50;45;20;10\n')


def prediction_and_target(predictions, ground_truth, scores=None):
    pred = np.asarray([r.as_array() for r in predictions], dtype=np.float32).reshape(-1, 5)
    gt = np.asarray([r.as_array() for r in ground_truth], dtype=np.float32).reshape(-1, 5)
    prediction = dict(grasps=pred, boxes=np.asarray([r.to_bbox() for r in predictions]).reshape(-1, 4),
                      scores=np.asarray(scores if scores is not None else [.99] * len(pred)))
    target = dict(grasps=gt, boxes=np.asarray([r.to_bbox() for r in ground_truth]).reshape(-1, 4))
    return prediction, target


class GeometryTests(unittest.TestCase):
    def test_corners_round_trip_and_edge_order(self):
        for angle in (-89, -30, 0, 30, 89, 179):
            for size in ((100, 20), (20, 100)):
                with self.subTest(angle=angle, size=size):
                    original = rectangle(angle, size=size)
                    result = GraspRectangle.from_corners(original.to_corners())
                    self.assertAlmostEqual(original.intersection_over_union(result), 1., places=5)
                    self.assertAlmostEqual(original.angle_error(result), 0., places=4)
                    self.assertAlmostEqual(result.width, size[0], places=4)

    def test_pi_periodicity(self):
        self.assertAlmostEqual(rectangle(1).angle_error(rectangle(179)), 2.)
        self.assertAlmostEqual(rectangle(30).angle_error(rectangle(210)), 0.)
        for degrees in (0, 30, 90, 179, 180):
            delta = math.radians(degrees)
            self.assertAlmostEqual(1 - math.cos(delta) ** 2, math.sin(delta) ** 2)

    def test_polygon_iou_does_not_substitute_aabb(self):
        a = rectangle(30, size=(100, 10))
        b = rectangle(59.9, size=(100, 10))
        self.assertGreater(aabb_iou(a.to_bbox(), b.to_bbox()), .25)
        self.assertLess(a.intersection_over_union(b), .25)
        prediction, target = prediction_and_target([a], [b])
        self.assertFalse(score_image(prediction, target)['success'])

    def test_perfect_rotated_prediction_has_perfect_overlap(self):
        grasp = rectangle(30)
        record = score_image(*prediction_and_target([grasp], [grasp]))
        self.assertTrue(record['success'])
        self.assertAlmostEqual(record['sample_ious'][0], 1., places=5)

    def test_any_valid_annotation_matches(self):
        pred = rectangle(0, center=(200, 200), size=(100, 100))
        gts = [rectangle(90, center=(200, 200), size=(100, 100)),
               rectangle(0, center=(220, 200), size=(100, 100))]
        record = score_image(*prediction_and_target([pred], gts))
        self.assertTrue(record['success'])
        self.assertEqual(record['true_positives'], 1)
        self.assertAlmostEqual(record['sample_ious'][0], 2 / 3, places=5)

    def test_confidence_top_k_and_one_to_one_matching(self):
        correct, wrong = rectangle(), rectangle(90, center=(10, 10))
        pred, gt = prediction_and_target([wrong, correct, correct], [correct], [.9, .8, .7])
        self.assertFalse(score_image(pred, gt, top_k=1)['success'])
        record = score_image(pred, gt, top_k=2)
        self.assertTrue(record['success'])
        self.assertEqual(record['true_positives'], 1)
        self.assertEqual(record['false_positives'], 2)
        self.assertFalse(score_image(pred, gt, confidence_threshold=.95)['success'])

    def test_empty_predictions_and_targets(self):
        record = score_image(*prediction_and_target([], []))
        self.assertFalse(record['success'])
        self.assertEqual(record['true_positives'], 0)

    def test_identity_augmentation_preserves_geometry_and_inputs(self):
        config = AugmentationConfig(rotation_limit=0, crop_ratio_min=1, crop_ratio_max=1,
                                    zoom_min=1, zoom_max=1, output_size=(128, 96),
                                    brightness_limit=0, contrast_limit=0)
        rgb = np.zeros((96, 128, 3), np.uint8)
        depth = np.zeros((96, 128), np.uint16)
        grasp = AugRectangle(64, 48, 30, 40, 12)
        original = grasp.get_corners().copy()
        image, dep, grasps = GraspAugmenter(config).augment_sample(rgb, depth, [grasp])
        np.testing.assert_array_equal(image, rgb)
        np.testing.assert_array_equal(dep, depth)
        np.testing.assert_allclose(grasps[0].get_corners(), original, atol=1e-5)
        np.testing.assert_array_equal(grasp.get_corners(), original)

    def test_shared_rgb_depth_and_annotation_transform(self):
        rgb = np.zeros((96, 128, 3), np.uint8)
        depth = np.zeros((96, 128), np.uint16)
        grasp = AugRectangle(64, 48, 30, 40, 20)
        polygon = grasp.get_corners().astype(np.int32)
        cv2.fillConvexPoly(rgb, polygon, (255, 255, 255))
        cv2.fillConvexPoly(depth, polygon, 1000)
        random.seed(7)
        config = AugmentationConfig(output_size=(128, 96), brightness_limit=0, contrast_limit=0)
        image, dep, grasps = GraspAugmenter(config).augment_sample(rgb, depth, [grasp])
        self.assertEqual(len(grasps), 1)
        rgb_mask, depth_mask = image[:, :, 0] > 127, dep > 0
        overlap = (rgb_mask & depth_mask).sum() / (rgb_mask | depth_mask).sum()
        self.assertGreater(overlap, .95)
        expected = np.zeros_like(depth)
        cv2.fillConvexPoly(expected, grasps[0].get_corners().astype(np.int32), 1)
        overlap = ((expected > 0) & depth_mask).sum() / ((expected > 0) | depth_mask).sum()
        self.assertGreater(overlap, .85)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cornell = self.root / 'cornell'
        self.sources = [dict(type='original_cornell', path=self.cornell)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_negative_annotations_and_missing_negative_file(self):
        pos = write_cornell(self.cornell, 1)
        sample = discover_samples(self.sources)[0]
        self.assertEqual(len(load_grasps(sample)), 1)
        pos.with_name(pos.name.replace('cpos', 'cneg')).unlink()
        dataset = GraspDataset(self.sources)
        _, target = dataset[0]
        self.assertEqual(target['labels'].tolist(), [1])
        self.assertEqual(target['grasps'].shape, (1, 5))

    def test_jacquard_square_is_letterboxed_with_unchanged_angle(self):
        jac = self.root / 'jac'
        write_jacquard(jac, 'objectA', 0)
        dataset = GraspDataset([dict(type='original_jacquard', path=jac)])
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 480, 640))
        np.testing.assert_allclose(target['grasps'][0].numpy(), [320, 240, 96, 48, math.pi / 4], atol=1e-5)

    def test_augmented_samples_keep_provenance_and_quality(self):
        write_cornell(self.cornell, 1)
        source = discover_samples(self.sources)[0]
        aug = self.root / 'aug'
        good, bad = AugRectangle(64, 48, 30, 40, 12), AugRectangle(64, 48, 80, 40, 12, quality=0)
        save_augmented_data(aug, 'sample', 0, np.zeros((96, 128, 3), np.uint8),
                            np.zeros((96, 128), np.uint16), [good, bad], 'cornell', source['sample_id'])
        samples = discover_samples([dict(type='augmented_cornell', path=aug)])
        grasps = load_grasps(samples[0])
        self.assertEqual(len(grasps), 1)
        self.assertAlmostEqual(math.degrees(grasps[0, 4]), 30., places=4)
        self.assertEqual(samples[0]['source_id'], source['sample_id'])

    def test_stale_augmentation_is_rejected(self):
        ann = self.root / 'aug' / 'annotations'
        ann.mkdir(parents=True)
        (ann / 'old.json').write_text(json.dumps(dict(grasps=[])))
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            discover_samples([dict(type='augmented_cornell', path=ann.parent)])

    def test_original_group_split_excludes_validation_augmentations_and_caps(self):
        for number in range(4):
            write_cornell(self.cornell, number, count=3)
        originals = discover_samples(self.sources)
        aug = self.root / 'aug'
        for index, original in enumerate(originals):
            save_augmented_data(aug, f'sample{index}', 0, np.zeros((96, 128, 3), np.uint8),
                                np.zeros((96, 128), np.uint16), [AugRectangle(64, 48, 30, 40, 12)],
                                'cornell', source_id=original['sample_id'])
        dataset = GraspDataset(self.sources + [dict(type='augmented_cornell', path=aug)], max_grasps_per_image=1)
        train, val, manifest = build_training_split(dataset, .5, self.root / 'models')
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 2)
        for index in train.indices:
            sample = dataset.samples[index]
            identity = sample.get('source_id', sample['sample_id'])
            self.assertIn(identity, manifest['train_ids'])
        self.assertTrue(all(val.dataset.samples[i]['format'] != 'json' for i in val.indices))
        self.assertEqual(len(train[0][1]['grasps']), 1)
        self.assertEqual(len(val[0][1]['grasps']), 3)
        # Reusing the same manifest is independent of augmentation enumeration.
        self.assertEqual(create_split(originals, .5), manifest)

    def test_jacquard_object_views_never_cross_partitions(self):
        jac = self.root / 'jac'
        for obj in ('objectA', 'objectB', 'objectC'):
            for view in (0, 1):
                write_jacquard(jac, obj, view)
        samples = discover_samples([dict(type='original_jacquard', path=jac)])
        manifest = create_split(samples, .5)
        train = {samples[i]['group_id'] for i in select_indices(samples, manifest, 'train')}
        val = {samples[i]['group_id'] for i in select_indices(samples, manifest, 'val')}
        self.assertFalse(train & val)
        validate_split(manifest, samples)
        manifest['val_ids'].append(manifest['train_ids'][0])
        with self.assertRaises(ValueError):
            validate_split(manifest, samples)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_geometry_encode_decode_preserves_actual_dimensions(self):
        g = torch.tensor([[60., 50., 80., 10., .7], [90., 70., 20., 60., 1.1]])
        proposals = torch.tensor([[0., 0., 120., 100.], [30., 20., 150., 120.]])
        decoded = decode_grasps(encode_grasps(g, proposals), proposals)
        torch.testing.assert_close(decoded, g[:, :4])

    def test_transform_and_postprocess_invert_letterbox(self):
        transform = GraspRCNNTransform([0, 0, 0], [1, 1, 1])
        target = dict(boxes=torch.tensor([[30., 40., 70., 60.]]),
                      grasps=torch.tensor([[50., 50., 40., 20., .7]]))
        original = {k: v.clone() for k, v in target.items()}
        image, transformed = transform.resize(torch.rand(3, 100, 100), target)
        self.assertEqual(tuple(image.shape[-2:]), (480, 640))
        self.assertAlmostEqual(float(transformed['grasps'][0, 4]), .7, places=6)
        transform.eval()
        result = transform.postprocess([transformed], [(480, 640)], [(100, 100)])[0]
        for key in original:
            torch.testing.assert_close(result[key], original[key])

    def test_legacy_warm_start_keeps_detector_and_reinitializes_grasp_heads(self):
        model = nn.Module()
        model.roi_heads = nn.Module()
        model.roi_heads.box_predictor = OrientedFastRCNNPredictor(8, 2)
        state = {key: torch.full_like(value, .25) for key, value in model.state_dict().items()
                 if '.angle_pred.' not in key and '.grasp_pred.' not in key}
        checkpoint = dict(model_state_dict=state)
        with self.assertRaisesRegex(ValueError, 'Warm-start'):
            load_training_weights(model, checkpoint)
        load_training_weights(model, checkpoint, allow_legacy=True)
        torch.testing.assert_close(model.roi_heads.box_predictor.cls_score.weight,
                                   torch.full_like(model.roi_heads.box_predictor.cls_score.weight, .25))
        self.assertEqual(float(model.roi_heads.box_predictor.grasp_pred.weight.detach().abs().sum()), 0.)

    def test_both_real_backbones_forward_backward_and_empty_targets(self):
        from train import get_model as resnet50
        from train_resnet_18 import get_model as resnet18
        for builder in (resnet50, resnet18):
            with self.subTest(backbone=builder.__module__):
                torch.manual_seed(123)
                model = builder(2, pretrained=False)
                from torchvision.ops.misc import FrozenBatchNorm2d
                self.assertIsInstance(model.backbone.body.bn1, FrozenBatchNorm2d)
                model.transform.output_size = (96, 128)
                model.rpn._pre_nms_top_n = dict(training=32, testing=32)
                model.rpn._post_nms_top_n = dict(training=16, testing=16)
                model.roi_heads.fg_bg_sampler.batch_size_per_image = 16
                grasp = rectangle()
                target = dict(grasps=torch.tensor(grasp.as_array()[None]),
                              boxes=torch.tensor([grasp.to_bbox()], dtype=torch.float32),
                              angles=torch.tensor([grasp.angle]), labels=torch.ones(1, dtype=torch.int64))
                empty = dict(grasps=torch.empty((0, 5)), boxes=torch.empty((0, 4)),
                             angles=torch.empty(0), labels=torch.empty(0, dtype=torch.int64))
                images = [torch.rand(3, 96, 128), torch.rand(3, 96, 128)]
                model.train()
                losses = model(images, [target, empty])
                self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
                self.assertIn('loss_grasp_geometry', losses)
                sum(losses.values()).backward()
                for head in (model.roi_heads.box_predictor.grasp_pred,
                             model.roi_heads.box_predictor.angle_pred[-1]):
                    self.assertTrue(torch.isfinite(head.weight.grad).all())
                    self.assertGreater(float(head.weight.grad.abs().sum()), 0.)
                model.eval()
                with torch.no_grad():
                    result = model(images)[0]
                self.assertEqual(result['grasps'].shape, (len(result['boxes']), 5))
                torch.testing.assert_close(result['angles'], result['grasps'][:, 4])
                self.assertTrue((result['grasps'][:, 2:4] > 0).all())
                # No foreground in the entire batch must still yield a finite backward pass.
                model.zero_grad()
                model.train()
                losses = model([images[0]], [empty])
                self.assertEqual(float(losses['loss_angle'].detach()), 0.)
                self.assertEqual(float(losses['loss_grasp_geometry'].detach()), 0.)
                sum(losses.values()).backward()


class EvaluationTests(unittest.TestCase):
    def test_training_checkpoint_and_standalone_evaluation_lifecycle(self):
        import train
        from grasp_evaluation import GraspEvaluator
        from oriented_model import OrientedRoIHeads, install_grasp_transform
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        from torchvision.ops import MultiScaleRoIAlign

        def tiny_detector(*args, **kwargs):
            backbone = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU())
            backbone.out_channels = 8
            model = FasterRCNN(backbone, num_classes=2,
                               rpn_anchor_generator=AnchorGenerator(((16, 32),), ((.5, 1., 2.),)),
                               box_roi_pool=MultiScaleRoIAlign(['0'], 2, 2),
                               rpn_pre_nms_top_n_train=16, rpn_post_nms_top_n_train=8,
                               rpn_pre_nms_top_n_test=16, rpn_post_nms_top_n_test=8,
                               box_batch_size_per_image=8)
            old = model.roi_heads
            model.roi_heads = OrientedRoIHeads(
                old.box_roi_pool, old.box_head, OrientedFastRCNNPredictor(1024, 2),
                old.proposal_matcher.high_threshold, old.proposal_matcher.low_threshold,
                old.fg_bg_sampler.batch_size_per_image, old.fg_bg_sampler.positive_fraction,
                old.box_coder.weights, old.score_thresh, old.nms_thresh, old.detections_per_img)
            install_grasp_transform(model)
            model.transform.output_size = (96, 128)
            return model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number in range(4):
                write_cornell(root / 'cornell', number)
            samples = discover_samples([dict(type='original_cornell', path=root / 'cornell')])
            manifest = create_split(samples, .5)
            initial = root / 'initial.pth'
            torch.save(dict(**checkpoint_info(manifest), model_state_dict=tiny_detector().state_dict()), initial)
            arguments = ['train.py', '--resume', str(initial), '--finetune_angle_head',
                         '--cornell_path', str(root / 'cornell'), '--jacquard_path', str(root / 'absent'),
                         '--augmented_cornell_path', str(root / 'absent'), '--num_epochs', '1',
                         '--batch_size', '1', '--num_workers', '0', '--device', 'cpu',
                         '--optimizer', 'adamw', '--scheduler', 'cosine', '--output_dir', str(root / 'models')]
            with mock.patch.object(train, 'get_model', tiny_detector), mock.patch('sys.argv', arguments):
                train.main()
            checkpoint = root / 'models' / 'final_model.pth'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(saved['grasp_format'], GRASP_FORMAT)
            self.assertEqual(saved['split_manifest'], manifest)
            self.assertIn('scheduler_state_dict', saved)
            self.assertTrue((root / 'models' / 'best_model.pth').exists())
            self.assertTrue((root / 'models' / 'flops.jsonl').exists())
            self.assertTrue((root / 'models' / 'training_curves.png').exists())
            evaluator = GraspEvaluator(checkpoint, root / 'cornell', 'original_cornell', tiny_detector, 'cpu')
            metrics, _ = evaluator.run()
            self.assertEqual(metrics['total_images'], 2)
            self.assertEqual(metrics['evaluation_scope'], 'saved_original_image_holdout')
            # Ordinary resume must infer the saved optimizer/scheduler, restore their
            # state, keep the split, and continue at the next epoch.
            arguments = ['train.py', '--resume', str(checkpoint), '--num_epochs', '2',
                         '--cornell_path', str(root / 'cornell'), '--jacquard_path', str(root / 'absent'),
                         '--augmented_cornell_path', str(root / 'absent'), '--batch_size', '1',
                         '--num_workers', '0', '--device', 'cpu', '--output_dir', str(root / 'models')]
            with mock.patch.object(train, 'get_model', tiny_detector), mock.patch('sys.argv', arguments):
                train.main()
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(saved['epoch'], 2)
            self.assertEqual(saved['optimizer_name'], 'adamw')
            self.assertEqual(saved['scheduler_name'], 'cosine')
            self.assertEqual(saved['scheduler_state_dict']['last_epoch'], 2)

    def test_standalone_evaluators_use_saved_holdout_and_reject_legacy(self):
        from grasp_evaluation import GraspEvaluator

        class PerfectModel(nn.Module):
            def forward(self, images):
                # Synthetic Cornell images are uniformly scaled from 128x96 to 640x480.
                grasp = rectangle().as_array()
                grasp[:4] *= 5
                rect = GraspRectangle.from_array(grasp)
                return [dict(grasps=torch.tensor(grasp[None]), boxes=torch.tensor([rect.to_bbox()]),
                             scores=torch.tensor([.99]), labels=torch.tensor([1]),
                             angles=torch.tensor([rect.angle])) for _ in images]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number in range(4):
                write_cornell(root / 'cornell', number)
            samples = discover_samples([dict(type='original_cornell', path=root / 'cornell')])
            manifest = create_split(samples, .5)
            checkpoint = root / 'model.pth'
            torch.save(dict(**checkpoint_info(manifest), model_state_dict={}), checkpoint)
            evaluator = GraspEvaluator(checkpoint, root / 'cornell', 'original_cornell',
                                       lambda **kwargs: PerfectModel(), device='cpu')
            metrics, _ = evaluator.run()
            self.assertEqual(metrics['accuracy'], 1.)
            self.assertEqual(metrics['total_images'], 2)
            self.assertEqual({r['sample_id'] for r in metrics['detailed_results']}, set(manifest['val_ids']))
            self.assertEqual(metrics['evaluation_scope'], 'saved_original_image_holdout')
            torch.save(dict(model_state_dict={}), checkpoint)
            with self.assertRaisesRegex(ValueError, 'predates'):
                GraspEvaluator(checkpoint, root / 'cornell', 'original_cornell', lambda **kwargs: PerfectModel())


if __name__ == '__main__':
    unittest.main()
