"""Create a portable holdout manifest before generating Cornell augmentations."""
import argparse
import json
from pathlib import Path

from grasp_data import create_split, discover_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cornell_path', default='./cornell_dataset')
    parser.add_argument('--jacquard_path', default='./jacquard_dataset')
    parser.add_argument('--train_ratio', type=float, default=.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='./dataset_split.json')
    args = parser.parse_args()
    sources = [dict(type='original_cornell', path=args.cornell_path),
               dict(type='original_jacquard', path=args.jacquard_path)]
    manifest = create_split(discover_samples(sources), args.train_ratio, args.seed)
    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != manifest:
            raise ValueError('Existing split differs; choose a new output path to preserve the old holdout')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Saved {len(manifest['train_ids'])} training and {len(manifest['val_ids'])} validation images to {output}")


if __name__ == '__main__':
    main()
