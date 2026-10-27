"""The same oriented matching rules for training validation and standalone evaluation."""
import numpy as np

from grasp_geometry import GraspRectangle, aabb_iou, grasp_match


def score_image(prediction, target, confidence_threshold=.5, iou_threshold=.25,
                angle_threshold=30., top_k=1, compute_per_grasp=True):
    scores = np.asarray(prediction['scores'])
    keep = np.where(scores >= confidence_threshold)[0]
    keep = keep[np.argsort(-scores[keep], kind='stable')]
    grasps = np.asarray(prediction['grasps']).reshape(-1, 5)[keep]
    boxes = np.asarray(prediction['boxes']).reshape(-1, 4)[keep]
    gt = np.asarray(target['grasps']).reshape(-1, 5)
    gt_boxes = np.asarray(target['boxes']).reshape(-1, 4)
    available = list(range(len(gt)))
    true_positives, ious, errors = 0, [], []
    for grasp in grasps if compute_per_grasp else []:
        index, iou, error = grasp_match(grasp, gt[available], iou_threshold, angle_threshold)
        if index >= 0:
            true_positives += 1
            available.pop(index)
            ious.append(iou)
            errors.append(error)
    success, aabb_success, aabb_angle_success = False, False, False
    for box, grasp in zip(boxes[:top_k], grasps[:top_k]):
        success |= grasp_match(grasp, gt, iou_threshold, angle_threshold)[0] >= 0
        pred_rect = GraspRectangle.from_array(grasp)
        for gt_box, gt_grasp in zip(gt_boxes, gt):
            localized = aabb_iou(box, gt_box) > iou_threshold
            aabb_success |= localized
            aabb_angle_success |= localized and pred_rect.angle_error(
                GraspRectangle.from_array(gt_grasp)) <= angle_threshold + 1e-5
    return dict(success=bool(success), aabb_success=bool(aabb_success),
                aabb_angle_success=bool(aabb_angle_success), true_positives=true_positives,
                false_positives=len(grasps) - true_positives,
                false_negatives=len(gt) - true_positives,
                num_predictions=len(grasps), num_gt_grasps=len(gt),
                sample_ious=ious, sample_angle_errors=errors,
                best_score=float(scores[keep[0]]) if len(keep) else 0.0,
                any_prediction=bool(len(scores)), per_grasp_metrics_computed=compute_per_grasp)


def summarize(records, confidence_threshold=.5, iou_threshold=.25, angle_threshold=30., top_k=1):
    total = len(records)
    tp = sum(r['true_positives'] for r in records)
    fp = sum(r['false_positives'] for r in records)
    fn = sum(r['false_negatives'] for r in records)
    correct = sum(r['success'] for r in records)
    localized = sum(r['aabb_success'] for r in records)
    localized_angle = sum(r['aabb_angle_success'] for r in records)
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    ious = [v for r in records for v in r['sample_ious']]
    angles = [v for r in records for v in r['sample_angle_errors']]
    max_scores = np.asarray([r['best_score'] for r in records] or [0.])
    result = dict(accuracy=correct / max(1, total), correct_predictions=correct, total_images=total,
                aabb_accuracy=localized / max(1, total),
                angle_success_given_aabb=localized_angle / max(1, localized),
                aabb_angle_accuracy=localized_angle / max(1, total),
                precision=precision, recall=recall,
                f1_score=2 * precision * recall / max(1e-12, precision + recall),
                true_positives=tp, false_positives=fp, false_negatives=fn,
                total_gt_grasps=tp + fn, total_pred_grasps=tp + fp, total_predictions=tp + fp,
                mean_iou=float(np.mean(ious)) if ious else 0.,
                mean_angle_error=float(np.mean(angles)) if angles else 0.,
                ious=ious, angle_errors=angles, detailed_results=records,
                num_samples=total, samples_evaluated=total,
                images_with_any_pred=sum(r['any_prediction'] for r in records),
                images_with_pred_at_conf=sum(r['num_predictions'] > 0 for r in records),
                score_mean=float(max_scores.mean()), score_median=float(np.median(max_scores)),
                score_p90=float(np.percentile(max_scores, 90)), score_max=float(max_scores.max()),
                confidence_threshold=confidence_threshold, iou_threshold=iou_threshold,
                angle_threshold=angle_threshold, top_k=top_k,
                metric='rotated_polygon_iou_and_pi_periodic_angle')
    if not all(r['per_grasp_metrics_computed'] for r in records):
        # Epoch validation needs top-k image success, not quadratic per-grasp
        # matching against hundreds/thousands of Jacquard annotations.
        for key in ('precision', 'recall', 'f1_score', 'true_positives', 'false_positives',
                    'false_negatives', 'mean_iou', 'mean_angle_error'):
            result[key] = None
        result['total_gt_grasps'] = sum(r['num_gt_grasps'] for r in records)
        result['total_pred_grasps'] = result['total_predictions'] = sum(r['num_predictions'] for r in records)
    result['per_grasp_metrics_computed'] = all(r['per_grasp_metrics_computed'] for r in records)
    return result
