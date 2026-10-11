"""Save repeated augmented views of one training image and its plain baseline.

This check needs only Pillow, NumPy and PyYAML, and uses the same augmentation
function as KITTIDataset.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from PIL import Image, ImageDraw
from utils.transforms import TrainingAugment, letterbox


def save_preview(rgb, targets, ignores, names, path):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for cls, xc, yc, bw, bh in targets:
        box = ((xc-bw/2)*width, (yc-bh/2)*height, (xc+bw/2)*width, (yc+bh/2)*height)
        draw.rectangle(box, outline='lime', width=2)
        draw.text((max(0, box[0]), max(0, box[1]-12)), names[int(cls)], fill='lime',
                  stroke_width=1, stroke_fill='black')
    for box in ignores:
        box = box*np.array([width, height, width, height])
        draw.rectangle(tuple(box), outline='white', width=2)
        draw.text((box[0], box[1]), 'Ignore', fill='white', stroke_width=1, stroke_fill='black')
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/kitti.yaml')
    parser.add_argument('--split-dir', type=Path, default=Path('splits/kitti'))
    parser.add_argument('--image-id', help='Defaults to first ID in train.txt')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('outputs/augmentation_checks'))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config['dataset_root'])
    ids = (args.split_dir/'train.txt').read_text().split()
    stem = args.image_id or ids[0]
    if stem not in ids:
        raise ValueError('Choose an image ID from train.txt')
    targets = np.asarray([list(map(float, l.split())) for l in
                          (root/'prepared/labels'/f'{stem}.txt').read_text().splitlines() if l.strip()],
                         dtype=np.float32).reshape(-1, 5)
    ignores = np.asarray(json.loads((root/'prepared/ignore'/f'{stem}.json').read_text()),
                         dtype=np.float32).reshape(-1, 4)
    with Image.open(root/'training/image_2'/f'{stem}.png') as source:
        image = source.convert('RGB')
    args.output = args.output/stem
    args.output.mkdir(parents=True, exist_ok=True)
    size = int(config.get('image_size', 416))
    rgb, t, ig, _ = letterbox(image, targets, ignores, size)
    save_preview(rgb, t, ig, config['names'], args.output/'original_preprocessed.png')
    settings = config.get('augmentation', {}) or {}
    if settings.get('enabled', True) is False:
        print('Preview explicitly demonstrates augmentation although training augmentation is disabled.')
    augment = TrainingAugment(settings)
    rng = random.Random(int(config.get('seed', 42)))
    records = []
    for index in range(args.samples):
        transformed, t, ig, info = augment(image, targets, ignores, rng)
        rgb, t, ig, _ = letterbox(transformed, t, ig, size)
        name = f'augmented_{index+1:02d}.png'
        save_preview(rgb, t, ig, config['names'], args.output/name)
        records.append(dict(file=name, targets=len(t), ignore_boxes=len(ig), transforms=info))
    (args.output/'transforms.json').write_text(json.dumps(records, indent=2))
    print(f'Saved baseline + {args.samples} augmented views to {args.output}')


if __name__ == '__main__':
    main()
