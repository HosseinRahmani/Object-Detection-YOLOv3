"""Create KITTI split ID lists and preview converted ground-truth boxes.

Run from the project root. Images and labels are never moved or copied.
Optional --groups JSON maps every image stem to its source sequence ID.
Without sequence metadata, seeded random image splits cannot rule out leakage.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_boxes(path, class_count):
    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'{path}:{line_number}: expected five finite values')
        cls, xc, yc, bw, bh = values
        if not cls.is_integer() or not 0 <= cls < class_count:
            raise ValueError(f'{path}:{line_number}: invalid class ID')
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f'{path}:{line_number}: invalid normalized box')
        if min(xc-bw/2, yc-bh/2) < -1e-7 or max(xc+bw/2, yc+bh/2) > 1+1e-7:
            raise ValueError(f'{path}:{line_number}: box extends outside image')
        boxes.append((int(cls), xc, yc, bw, bh))
    return boxes


def make_splits(ids, ratios, seed, groups=None):
    rng = random.Random(seed)
    if groups is None:
        shuffled = sorted(ids)
        rng.shuffle(shuffled)
        n_train, n_val = int(len(ids)*ratios[0]), int(len(ids)*ratios[1])
        result = dict(train=shuffled[:n_train], val=shuffled[n_train:n_train+n_val],
                      test=shuffled[n_train+n_val:])
    else:
        if any(i not in groups for i in ids):
            raise ValueError('Group mapping must include every image ID')
        members = defaultdict(list)
        for i in sorted(ids):
            group = groups[i]
            if not isinstance(group, (str, int)):
                raise ValueError('Group IDs must be strings or integers')
            members[group].append(i)
        batches = list(members.values())
        if len(batches) < 3:
            raise ValueError('At least three source groups are required')
        rng.shuffle(batches)
        batches.sort(key=len, reverse=True)
        result = {s: [] for s in ('train', 'val', 'test')}
        targets = dict(zip(result, (len(ids)*r for r in ratios)))
        for batch in batches:
            split = max(result, key=lambda s: (targets[s]-len(result[s]))/targets[s])
            result[split].extend(batch)
    if any(not subset for subset in result.values()):
        raise ValueError('Empty split; provide more images/groups or change ratios')
    assert len(set().union(*(set(v) for v in result.values()))) == len(ids)
    assert sum(map(len, result.values())) == len(ids)
    return {s: sorted(v) for s, v in result.items()}


def draw_preview(image_path, boxes, ignores, names, destination):
    from PIL import Image, ImageDraw
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(image)
    palette = ['#ff4444', '#44dd44', '#4488ff', '#ffcc00',
               '#ee55ff', '#00dddd', '#ff8800', '#aabbff']
    for cls, xc, yc, bw, bh in boxes:
        box = ((xc-bw/2)*width, (yc-bh/2)*height,
               (xc+bw/2)*width, (yc+bh/2)*height)
        color = palette[cls % len(palette)]
        draw.rectangle(box, outline=color, width=3)
        x, y = max(0, box[0]), max(0, box[1]-14)
        text = f'{cls}: {names[cls]}'
        background = draw.textbbox((x, y), text)
        draw.rectangle(background, fill='black')
        draw.text((x, y), text, fill=color)
    for box in ignores:
        draw.rectangle(box, outline='white', width=2)
        draw.text((max(0, box[0]), max(0, box[1])), 'DontCare', fill='white', stroke_width=1, stroke_fill='black')
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/kitti.yaml')
    parser.add_argument('--ratios', type=float, nargs=3, default=[.7, .15, .15])
    parser.add_argument('--samples', type=int, default=10, help='Previews per split')
    parser.add_argument('--groups', type=Path, help='Optional image-ID to sequence-ID JSON')
    parser.add_argument('--split-dir', type=Path, default=Path('splits/kitti'))
    parser.add_argument('--preview-dir', type=Path, default=Path('outputs/dataset_checks'))
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace existing split files')
    args = parser.parse_args()
    if any(not math.isfinite(r) or r <= 0 for r in args.ratios) or not math.isclose(sum(args.ratios), 1):
        parser.error('Three positive ratios must sum to 1')
    if args.samples < 0:
        parser.error('--samples cannot be negative')
    config = yaml.safe_load(Path(args.config).read_text())
    root, names = Path(config['dataset_root']), config['names']
    if not names or len(set(names)) != len(names) or 'DontCare' in names:
        raise ValueError('Expected unique target class names excluding DontCare')
    images = {p.stem: p for p in sorted((root/'training/image_2').glob('*.png'))}
    if not images:
        raise FileNotFoundError(f'No training PNG images under {root}')
    labels = root/'prepared/labels'
    ignore_dir = root/'prepared/ignore'
    boxes, ignores = {}, {}
    for stem in images:
        boxes[stem] = read_boxes(labels/f'{stem}.txt', len(names))
        ignores[stem] = json.loads((ignore_dir/f'{stem}.json').read_text())
        for box in ignores[stem]:
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'Invalid ignore box for {stem}')
    groups = json.loads(args.groups.read_text()) if args.groups else None
    seed = int(config.get('seed', 42))
    splits = make_splits(list(images), args.ratios, seed, groups)
    targets = [args.split_dir/f'{s}.txt' for s in splits] + [args.split_dir/'summary.json']
    if not args.overwrite and any(p.exists() for p in targets):
        raise FileExistsError('Split files already exist; use --overwrite only to deliberately replace them')
    summary = dict(seed=seed, requested_ratios=args.ratios,
                   split_method='sequence groups' if groups else 'random images',
                   names=names, splits={})
    for split, ids in splits.items():
        counts = Counter(b[0] for i in ids for b in boxes[i])
        stats = dict(images=len(ids), class_counts={n: counts[c] for c, n in enumerate(names)},
                     empty_target_images=sum(not boxes[i] for i in ids),
                     dontcare_boxes=sum(len(ignores[i]) for i in ids))
        summary['splits'][split] = stats
        missing = [n for c, n in enumerate(names) if not counts[c]]
        if missing:
            print(f'Warning: {split} has no examples of: {", ".join(missing)}')
        # Sample some examples of each class, then fill remaining slots randomly.
        shuffled = ids.copy()
        random.Random(seed).shuffle(shuffled)
        selected = []
        for cls in range(len(names)):
            candidate = next((i for i in shuffled if i not in selected and any(b[0] == cls for b in boxes[i])), None)
            if candidate and len(selected) < args.samples:
                selected.append(candidate)
        selected += [i for i in shuffled if i not in selected][:max(0, args.samples-len(selected))]
        for stem in selected:
            draw_preview(images[stem], boxes[stem], ignores[stem], names,
                         args.preview_dir/split/f'{stem}.png')
    args.split_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (args.split_dir/f'{split}.txt').write_text(''.join(i+'\n' for i in ids))
    (args.split_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Previews: {args.preview_dir}')
    if groups is None:
        print('Random image split: source-sequence leakage has not been ruled out.')


if __name__ == '__main__':
    main()
