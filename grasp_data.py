"""Dataset discovery, positive annotations and portable, grouped train/val manifests."""
import hashlib
import json
from pathlib import Path

import numpy as np

from grasp_geometry import GraspRectangle


def cornell_sample_id(pos_file, root):
    relative = Path(pos_file).relative_to(root).as_posix()
    return 'cornell/' + relative.removesuffix('cpos.txt')


def discover_samples(data_sources):
    samples = []
    for source in data_sources:
        root, kind = Path(source['path']), source['type']
        if not root.is_dir():
            continue
        if kind == 'original_cornell':
            for pos in sorted(root.rglob('*cpos.txt')):
                if 'backgrounds' in pos.relative_to(root).parts:
                    continue
                base = pos.name.removesuffix('cpos.txt')
                rgb = pos.with_name(base + 'r.png')
                if not rgb.exists():
                    continue
                sample_id = cornell_sample_id(pos, root)
                samples.append(dict(rgb_path=str(rgb), pos_grasps_path=str(pos),
                                    sample_id=sample_id, group_id=sample_id,
                                    source=kind, format='cornell'))
        elif kind == 'original_jacquard':
            for annotation in sorted(root.rglob('*_grasps.txt')):
                base = annotation.name.removesuffix('_grasps.txt')
                rgb = annotation.with_name(base + '_RGB.png')
                if not rgb.exists():
                    continue
                sample_id = 'jacquard/' + annotation.relative_to(root).as_posix().removesuffix('_grasps.txt')
                # All rendered views of one object stay in one partition.
                group_id = 'jacquard/' + annotation.parent.name
                samples.append(dict(rgb_path=str(rgb), grasps_path=str(annotation),
                                    sample_id=sample_id, group_id=group_id,
                                    source=kind, format='jacquard'))
        elif kind == 'augmented_cornell':
            for annotation in sorted((root / 'annotations').glob('*.json')):
                metadata = json.loads(annotation.read_text())
                if metadata.get('geometry_version') != 2 or not metadata.get('source_id'):
                    raise ValueError(f'{annotation}: regenerate augmentations with geometry_version=2 and source_id')
                rgb = root / 'rgb' / (annotation.stem + '.png')
                if not rgb.exists():
                    raise FileNotFoundError(rgb)
                source_id = metadata['source_id']
                samples.append(dict(rgb_path=str(rgb), annotation_path=str(annotation),
                                    sample_id='augmented/' + annotation.stem,
                                    source_id=source_id, group_id=source_id,
                                    source=kind, format='json'))
        else:
            raise ValueError(f'Unknown dataset type: {kind}')
    original_ids = [s['sample_id'] for s in samples if s['format'] != 'json']
    if len(original_ids) != len(set(original_ids)):
        raise ValueError('Duplicate original sample IDs; check overlapping dataset roots')
    return samples


def load_grasps(sample):
    """Return only successful, finite grasps in the shared image-angle convention."""
    rectangles = []
    if sample['format'] == 'cornell':
        lines = Path(sample['pos_grasps_path']).read_text().splitlines()
        for index in range(0, len(lines) - 3, 4):
            try:
                corners = np.array([[float(v) for v in line.split()]
                                    for line in lines[index:index + 4]])
                rectangles.append(GraspRectangle.from_corners(corners))
            except (ValueError, IndexError):
                continue
    elif sample['format'] == 'jacquard':
        for line in Path(sample['grasps_path']).read_text().splitlines():
            try:
                x, y, angle, width, height = map(float, line.split(';'))
                rectangles.append(GraspRectangle(x, y, width, height, angle, 'jacquard'))
            except ValueError:
                continue
    elif sample['format'] == 'json':
        for grasp in json.loads(Path(sample['annotation_path']).read_text())['grasps']:
            if grasp.get('quality', 1.0) <= 0.5:
                continue
            try:
                rectangles.append(GraspRectangle.from_corners(grasp['corners']))
            except ValueError:
                continue
    else:
        raise ValueError(sample['format'])
    return np.asarray([r.as_array() for r in rectangles
                       if np.isfinite(r.as_array()).all() and min(r.width, r.height) > 0],
                      dtype=np.float32).reshape(-1, 5)


def create_split(samples, train_ratio=0.9, seed=42):
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must lie strictly between 0 and 1')
    originals = [s for s in samples if s['format'] != 'json']
    if not originals:
        raise ValueError('Original Cornell/Jacquard images are required to create a split')
    train_groups, val_groups = [], []
    # Stratify by dataset so one large Jacquard collection cannot swallow Cornell.
    for source in sorted({s['source'] for s in originals}):
        groups = sorted({s['group_id'] for s in originals if s['source'] == source})
        if len(groups) < 2:
            raise ValueError(f'{source}: need at least two independent image/object groups')
        rng = np.random.default_rng(seed)
        groups = list(np.asarray(groups)[rng.permutation(len(groups))])
        count = min(len(groups) - 1, max(1, int(len(groups) * train_ratio)))
        train_groups.extend(groups[:count])
        val_groups.extend(groups[count:])
    training = set(train_groups)
    validation = set(val_groups)
    return dict(version=2, seed=seed, train_ratio=train_ratio,
                train_ids=sorted(s['sample_id'] for s in originals if s['group_id'] in training),
                val_ids=sorted(s['sample_id'] for s in originals if s['group_id'] in validation),
                train_groups=sorted(training), val_groups=sorted(validation))


def validate_split(manifest, samples=None):
    if manifest.get('version') != 2:
        raise ValueError('Unsupported split manifest version; generate a new split')
    train, val = set(manifest['train_ids']), set(manifest['val_ids'])
    if not train or not val or train & val:
        raise ValueError('Split must contain nonempty, disjoint train and validation IDs')
    if set(manifest['train_groups']) & set(manifest['val_groups']):
        raise ValueError('An image/object group appears in both partitions')
    if samples is not None:
        originals = {s['sample_id']: s for s in samples if s['format'] != 'json'}
        if train | val != set(originals):
            raise ValueError('Split IDs do not match the current original datasets; use the same dataset roots')
        for partition in ('train', 'val'):
            groups = set(manifest[partition + '_groups'])
            if any(originals[key]['group_id'] not in groups for key in manifest[partition + '_ids']):
                raise ValueError('Sample group differs from its split partition')
    return manifest


def split_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def read_split(path):
    return validate_split(json.loads(Path(path).read_text()))


def select_indices(samples, manifest, partition):
    if partition not in ('train', 'val'):
        raise ValueError('partition must be train or val')
    allowed = set(manifest[partition + '_ids'])
    indices = []
    for index, sample in enumerate(samples):
        if sample['format'] == 'json':
            if partition == 'train' and sample['source_id'] in allowed:
                indices.append(index)
        elif sample['sample_id'] in allowed:
            indices.append(index)
    return indices
