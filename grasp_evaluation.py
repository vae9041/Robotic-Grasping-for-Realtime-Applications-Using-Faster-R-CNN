"""Standalone evaluation against the exact saved original-image holdout."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from grasp_data import read_split, split_digest
from grasp_geometry import GraspRectangle
from grasp_metrics import score_image, summarize
from oriented_model import GRASP_FORMAT, load_training_weights
from train import GraspDataset


class GraspEvaluator:
    def __init__(self, model_path, dataset_path, dataset_type, model_builder,
                 device='auto', split_manifest=None, allow_full_dataset=False):
        if device in (None, 'auto'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if checkpoint.get('grasp_format') != GRASP_FORMAT:
            raise ValueError('Checkpoint predates corrected grasp geometry; warm-start and retrain its grasp heads first')
        self.model = model_builder(num_classes=2, freeze_backbone=False, pretrained=False)
        if checkpoint.get('structured_pruning_meta'):
            from structured_pruning_resnet_18 import load_structured_pruned_state_dict
            load_structured_pruned_state_dict(self.model, checkpoint, self.device)
        else:
            load_training_weights(self.model, checkpoint)
        self.model.to(self.device).eval()
        self.dataset = GraspDataset([dict(type=dataset_type, path=dataset_path)], max_grasps_per_image=None)
        manifest = read_split(split_manifest) if split_manifest else checkpoint.get('split_manifest')
        if split_manifest and checkpoint.get('split_manifest'):
            if split_digest(manifest) != split_digest(checkpoint['split_manifest']):
                raise ValueError('Evaluation split differs from the checkpoint training split')
        if allow_full_dataset:
            self.evaluation_scope = 'full_dataset_diagnostic_includes_training_images'
            self.split_digest = None
        else:
            if manifest is None:
                raise ValueError('Evaluation requires the training split manifest; use --split_manifest')
            prefix = 'cornell/' if dataset_type == 'original_cornell' else 'jacquard/'
            allowed = {key for key in manifest['val_ids'] if key.startswith(prefix)}
            available = {sample['sample_id'] for sample in self.dataset.samples}
            if not allowed or not allowed <= available:
                raise ValueError('Validation IDs are empty or missing from this dataset root')
            self.dataset.samples = [s for s in self.dataset.samples if s['sample_id'] in allowed]
            self.evaluation_scope = ('saved_split_legacy_pretraining_unverified'
                                     if checkpoint.get('holdout_provenance') == 'unverified_legacy_warm_start'
                                     else 'saved_original_image_holdout')
            self.split_digest = split_digest(manifest)
        if not len(self.dataset):
            raise ValueError('No evaluation images found')
        self.samples = self.dataset.samples

    def run(self, confidence_threshold=.5, iou_threshold=.25, angle_threshold=30., max_samples=None, top_k=1):
        if top_k < 1 or not 0 <= angle_threshold <= 90:
            raise ValueError('top_k must be positive and angle_threshold must lie in [0, 90]')
        count = min(len(self.dataset), max_samples) if max_samples is not None else len(self.dataset)
        if count < 1:
            raise ValueError('max_samples must be positive')
        records, predictions = [], []
        with torch.no_grad():
            for index in tqdm(range(count), desc='Evaluating original holdout'):
                image, target = self.dataset[index]
                prediction = self.model([image.to(self.device)])[0]
                prediction = {key: value.cpu().numpy() for key, value in prediction.items()}
                target = {key: value.numpy() for key, value in target.items()}
                record = score_image(prediction, target, confidence_threshold, iou_threshold, angle_threshold, top_k)
                record['sample_id'] = self.samples[index]['sample_id']
                records.append(record)
                predictions.append(dict(sample_id=record['sample_id'], image=image.numpy(),
                                        prediction=prediction, target=target))
        metrics = summarize(records, confidence_threshold, iou_threshold, angle_threshold, top_k)
        metrics.update(evaluation_scope=self.evaluation_scope, split_digest=self.split_digest)
        return metrics, predictions

    def evaluate(self, **kwargs):
        return self.run(**kwargs)

    def evaluate_success_rate(self, **kwargs):
        return self.run(**kwargs)[0]

    def visualize_predictions(self, predictions, output_dir, num_samples=10, confidence_threshold=.5):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(predictions[:num_samples]):
            rgb = np.clip(item['image'].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for grasp in item['target']['grasps']:
                corners = GraspRectangle.from_array(grasp).to_corners().astype(np.int32)
                cv2.polylines(image, [corners], True, (0, 255, 0), 1)
            keep = item['prediction']['scores'] >= confidence_threshold
            for grasp in item['prediction']['grasps'][keep]:
                corners = GraspRectangle.from_array(grasp).to_corners().astype(np.int32)
                cv2.polylines(image, [corners], True, (0, 0, 255), 2)
            cv2.imwrite(str(output_dir / f'{index:05d}.png'), image)


def evaluation_cli(evaluator_class, dataset_name, argv=None):
    parser = argparse.ArgumentParser(description=f'Evaluate oriented grasps on the saved {dataset_name} holdout')
    parser.add_argument('--model_path', required=True)
    parser.add_argument(f'--{dataset_name}_path', default=f'./{dataset_name}_dataset')
    parser.add_argument('--split_manifest', help='Uses checkpoint-embedded split by default')
    parser.add_argument('--allow_full_dataset', action='store_true', help='Explicit diagnostic; includes training images')
    parser.add_argument('--confidence_threshold', type=float, default=.5)
    parser.add_argument('--iou_threshold', type=float, default=.25)
    parser.add_argument('--angle_threshold', type=float, default=30.)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--output_dir', default=f'./{dataset_name}_evaluation')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_visualizations', type=int, default=10)
    args = parser.parse_args(argv)
    evaluator = evaluator_class(args.model_path, getattr(args, dataset_name + '_path'), args.device,
                                split_manifest=args.split_manifest, allow_full_dataset=args.allow_full_dataset)
    metrics, predictions = evaluator.run(args.confidence_threshold, args.iou_threshold,
                                         args.angle_threshold, args.max_samples, args.top_k)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'evaluation_results.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f"Evaluation scope: {metrics['evaluation_scope']}")
    print(f"Original images: {metrics['total_images']}; top_k={args.top_k}")
    print(f"AABB success: {metrics['aabb_accuracy']:.2%}")
    print(f"Angle success given AABB localization: {metrics['angle_success_given_aabb']:.2%}")
    print(f"Rotated grasp success: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.4f}; recall: {metrics['recall']:.4f}")
    if args.visualize:
        evaluator.visualize_predictions(predictions, output_dir / 'visualizations',
                                         args.num_visualizations, args.confidence_threshold)
