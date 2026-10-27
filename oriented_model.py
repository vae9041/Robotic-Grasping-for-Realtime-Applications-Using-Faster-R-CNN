"""Faster R-CNN extensions for actual oriented grasp dimensions and periodic angles."""
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops

GRASP_FORMAT = 'image_xy_clockwise_radians_v2'
logger = logging.getLogger(__name__)


class GraspRCNNTransform(GeneralizedRCNNTransform):
    """One uniform scale plus padding, including the custom target/output fields."""
    def __init__(self, image_mean, image_std, output_size=(480, 640)):
        super().__init__(min(output_size), max(output_size), image_mean, image_std)
        self.output_size = output_size

    def _parameters_for(self, shape):
        height, width = shape
        out_h, out_w = self.output_size
        scale = min(out_w / width, out_h / height)
        return scale, (out_w - scale * width) / 2, (out_h - scale * height) / 2

    def resize(self, image, target=None):
        height, width = image.shape[-2:]
        scale, dx, dy = self._parameters_for((height, width))
        if tuple(image.shape[-2:]) != self.output_size:
            out_h, out_w = self.output_size
            y, x = torch.meshgrid(
                torch.arange(out_h, dtype=image.dtype, device=image.device),
                torch.arange(out_w, dtype=image.dtype, device=image.device), indexing='ij')
            grid = torch.stack((((x + .5 - dx) / scale) * (2 / width) - 1,
                                ((y + .5 - dy) / scale) * (2 / height) - 1), dim=-1)
            image = F.grid_sample(image[None], grid[None], align_corners=False)[0]
        if target is not None:
            target['boxes'] = target['boxes'] * scale + image.new_tensor([dx, dy, dx, dy])
            target['grasps'] = target['grasps'].clone()
            target['grasps'][:, :2] = target['grasps'][:, :2] * scale + image.new_tensor([dx, dy])
            target['grasps'][:, 2:4] *= scale
        return image, target

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for prediction, original_size in zip(result, original_image_sizes):
            scale, dx, dy = self._parameters_for(original_size)
            boxes = prediction['boxes']
            boxes = (boxes - boxes.new_tensor([dx, dy, dx, dy])) / scale
            prediction['boxes'] = box_ops.clip_boxes_to_image(boxes, original_size)
            grasps = prediction['grasps'].clone()
            grasps[:, :2] = (grasps[:, :2] - grasps.new_tensor([dx, dy])) / scale
            grasps[:, 2:4] /= scale
            prediction['grasps'] = grasps
        return result


def install_grasp_transform(model):
    old = model.transform
    # Preserve Faster R-CNN's effective training resolution rather than silently
    # reducing its usual 800-pixel short side to the dataset's 480-pixel canvas.
    scale = min(min(old.min_size) / 480, old.max_size / 640)
    output_size = (round(480 * scale), round(640 * scale))
    model.transform = GraspRCNNTransform(old.image_mean, old.image_std, output_size)
    return model


def encode_grasps(grasps, proposals):
    size = (proposals[:, 2:] - proposals[:, :2]).clamp(min=1e-6)
    center = (proposals[:, 2:] + proposals[:, :2]) / 2
    return torch.cat(((grasps[:, :2] - center) / size,
                      torch.log(grasps[:, 2:4].clamp(min=1e-6) / size)), dim=1)


def decode_grasps(encoded, proposals):
    size = (proposals[:, 2:] - proposals[:, :2]).clamp(min=1e-6)
    center = (proposals[:, 2:] + proposals[:, :2]) / 2
    return torch.cat((encoded[:, :2] * size + center,
                      encoded[:, 2:].clamp(max=math.log(1000 / 16)).exp() * size), dim=1)


class OrientedFastRCNNPredictor(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)
        self.grasp_pred = nn.Linear(in_features, 4)
        nn.init.zeros_(self.grasp_pred.weight)
        nn.init.zeros_(self.grasp_pred.bias)
        self.angle_pred = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, 2))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return (self.cls_score(x), self.bbox_pred(x),
                F.normalize(self.angle_pred(x), dim=1, eps=1e-6), self.grasp_pred(x))


class OrientedRoIHeads(RoIHeads):
    angle_loss_weight = 1.0

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            if targets is None:
                raise ValueError('Training requires oriented targets')
            proposals, matched, labels, regression_targets = self.select_training_samples(proposals, targets)
        features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        logits, box_regression, angle_vectors, geometry = self.box_predictor(features)
        if self.training:
            classification, box_loss = fastrcnn_loss(logits, box_regression, labels, regression_targets)
            predicted_angles, target_angles, predicted_geometry, target_geometry = [], [], [], []
            offset = 0
            for props, indices, classes, target in zip(proposals, matched, labels, targets):
                positive = torch.where(classes > 0)[0]
                gt = target['grasps'][indices[positive]]
                predicted_angles.append(angle_vectors[offset + positive])
                target_angles.append(torch.stack((gt[:, 4].sin(), gt[:, 4].cos()), dim=1))
                predicted_geometry.append(geometry[offset + positive])
                target_geometry.append(encode_grasps(gt, props[positive]))
                offset += len(props)
            pred_angle, gt_angle = torch.cat(predicted_angles), torch.cat(target_angles)
            pred_geometry, gt_geometry = torch.cat(predicted_geometry), torch.cat(target_geometry)
            if len(pred_angle):
                dot = (pred_angle * gt_angle).sum(dim=1).clamp(-1, 1)
                angle_loss = (1 - dot.square()).mean() * self.angle_loss_weight
                geometry_loss = F.smooth_l1_loss(pred_geometry, gt_geometry, beta=1 / 9)
            else:
                angle_loss = angle_vectors.sum() * 0
                geometry_loss = geometry.sum() * 0
            return [], dict(loss_classifier=classification, loss_box_reg=box_loss,
                            loss_angle=angle_loss, loss_grasp_geometry=geometry_loss)
        boxes, scores, classes, grasps = self.postprocess_detections(
            logits, box_regression, angle_vectors, geometry, proposals, image_shapes)
        return [dict(boxes=b, scores=s, labels=c, grasps=g, angles=g[:, 4])
                for b, s, c, g in zip(boxes, scores, classes, grasps)], {}

    def postprocess_detections(self, logits, box_regression, angle_vectors, geometry, proposals, image_shapes):
        counts = [len(p) for p in proposals]
        num_classes = logits.shape[1]
        boxes = self.box_coder.decode(box_regression, proposals).split(counts)
        scores = F.softmax(logits, dim=1).split(counts)
        angles = torch.atan2(angle_vectors[:, 0], angle_vectors[:, 1]).remainder(math.pi)
        geom = decode_grasps(geometry, torch.cat(proposals))
        grasps = torch.cat((geom, angles[:, None]), dim=1).split(counts)
        all_boxes, all_scores, all_classes, all_grasps = [], [], [], []
        for box, score, grasp, shape in zip(boxes, scores, grasps, image_shapes):
            box = box_ops.clip_boxes_to_image(box, shape)
            classes = torch.arange(num_classes, device=logits.device).expand_as(score)
            box, score, classes = box[:, 1:].reshape(-1, 4), score[:, 1:].flatten(), classes[:, 1:].flatten()
            grasp = grasp[:, None, :].expand(-1, num_classes - 1, -1).reshape(-1, 5)
            keep = torch.where(score > self.score_thresh)[0]
            box, score, classes, grasp = box[keep], score[keep], classes[keep], grasp[keep]
            keep = box_ops.remove_small_boxes(box, 1e-2)
            box, score, classes, grasp = box[keep], score[keep], classes[keep], grasp[keep]
            keep = box_ops.batched_nms(box, score, classes, self.nms_thresh)[:self.detections_per_img]
            all_boxes.append(box[keep])
            all_scores.append(score[keep])
            all_classes.append(classes[keep])
            all_grasps.append(grasp[keep])
        return all_boxes, all_scores, all_classes, all_grasps


def load_training_weights(model, checkpoint, allow_legacy=False):
    """Old box-only / pre-fix angle checkpoints are warm starts, not resumable grasps."""
    state = checkpoint.get('model_state_dict', checkpoint)
    current = model.state_dict()
    legacy = checkpoint.get('grasp_format') != GRASP_FORMAT
    if legacy and not allow_legacy:
        raise ValueError('This checkpoint predates corrected grasp geometry. Warm-start it with a fine-tuning flag.')
    if legacy:
        # The former angle convention and geometry targets are incompatible.
        state = {key: value for key, value in state.items()
                 if '.angle_pred.' not in key and '.grasp_pred.' not in key}
    incompatible = [key for key, value in state.items() if key in current and current[key].shape != value.shape]
    if incompatible:
        raise ValueError(f'Checkpoint tensor shapes differ: {incompatible}')
    result = model.load_state_dict(state, strict=not legacy)
    if legacy:
        allowed = ('.angle_pred.', '.grasp_pred.')
        unexpected_missing = [key for key in result.missing_keys if not any(part in key for part in allowed)]
        if unexpected_missing or result.unexpected_keys:
            raise ValueError(f'Checkpoint differs outside grasp heads: {result}')
        logger.info('Loaded detector weights; initialized new angle and grasp geometry heads')
    return result
