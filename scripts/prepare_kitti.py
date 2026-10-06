"""Convert KITTI 2D annotations; retain original labels for KITTI evaluation."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/kitti.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config['dataset_root'])
    names = config['names']
    if len(names) != len(set(names)) or 'DontCare' in names:
        raise ValueError('Class names must be unique and exclude DontCare')
    class_ids = {name: i for i, name in enumerate(names)}
    images = root / 'training/image_2'
    labels = root / 'training/label_2'
    image_paths = sorted(images.glob('*.png'))
    if not image_paths or not labels.is_dir():
        raise FileNotFoundError(f'Expected PNG images in {images} and labels in {labels}')
    # Validate all inputs before creating output files.
    converted = []
    counts = Counter()
    ignored = invalid = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Unreadable image: {image_path}')
        height, width = image.shape[:2]
        label_path = labels / f'{image_path.stem}.txt'
        rows, ignore_boxes = [], []
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 15:
                raise ValueError(f'{label_path}:{line_number}: expected 15 fields')
            name = fields[0]
            if name != 'DontCare' and name not in class_ids:
                raise ValueError(f'Unknown class {name!r} in {label_path}')
            box = list(map(float, fields[4:8]))
            if not all(math.isfinite(value) for value in box):
                raise ValueError(f'Non-finite box in {label_path}')
            x1, y1, x2, y2 = box
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                invalid += 1
                continue
            if name == 'DontCare':
                ignore_boxes.append([x1, y1, x2, y2])
                ignored += 1
                continue
            xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            rows.append(f'{class_ids[name]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}')
            counts[name] += 1
        converted.append((image_path.stem, rows, ignore_boxes))
    output = root / 'prepared'
    (output / 'labels').mkdir(parents=True, exist_ok=True)
    (output / 'ignore').mkdir(parents=True, exist_ok=True)
    for stem, rows, boxes in converted:
        (output / 'labels' / f'{stem}.txt').write_text(''.join(row + '\n' for row in rows))
        # Ignore boxes use absolute xyxy coordinates, unlike YOLO labels.
        (output / 'ignore' / f'{stem}.json').write_text(json.dumps(boxes))
    summary = dict(images=len(converted), class_counts={n: counts[n] for n in names},
                   dontcare_boxes=ignored, skipped_invalid_boxes=invalid,
                   images_without_targets=sum(not rows for _, rows, _ in converted))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
