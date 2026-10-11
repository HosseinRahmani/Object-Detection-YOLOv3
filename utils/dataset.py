"""KITTI loader for split IDs produced by split_and_inspect_kitti.py."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .transforms import letterbox, normalize_image, TrainingAugment


class KITTIDataset(Dataset):
    def __init__(self, config_path='configs/kitti.yaml', split='train',
                 split_dir='splits/kitti', augment=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        config = yaml.safe_load(Path(config_path).read_text())
        self.root = Path(config['dataset_root'])
        self.names = config['names']
        self.image_size = int(config.get('image_size', 416))
        settings = config.get('augmentation', {}) or {}
        requested = settings.get('enabled', True) if augment is None else augment
        self.augment = TrainingAugment(settings) if split == 'train' and requested else None
        self.seed = int(config.get('seed', 42))
        self._rng = None
        self._rng_seed = None
        if not self.names or len(set(self.names)) != len(self.names) or 'DontCare' in self.names:
            raise ValueError('Expected unique target class names excluding DontCare')
        if self.image_size <= 0 or self.image_size % 32:
            raise ValueError('image_size must be a positive multiple of 32')
        self.ids = [s.strip() for s in (Path(split_dir)/f'{split}.txt').read_text().splitlines() if s.strip()]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('Split must contain nonempty, unique image IDs')
        for stem in self.ids:
            if '/' in stem or '\\' in stem or stem in ('.', '..'):
                raise ValueError(f'Invalid image ID: {stem}')
            for path in self.paths(stem):
                if not path.is_file():
                    raise FileNotFoundError(path)

    def paths(self, stem):
        return (self.root/'training/image_2'/f'{stem}.png',
                self.root/'prepared/labels'/f'{stem}.txt',
                self.root/'prepared/ignore'/f'{stem}.json')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stem = self.ids[index]
        image_path, label_path, ignore_path = self.paths(stem)
        rows = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5:
                raise ValueError(f'Expected five values per label in {label_path}')
            rows.append(values)
        targets = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        if len(targets):
            cls = targets[:, 0]
            if not np.isfinite(targets).all() or not ((cls == np.floor(cls)) & (cls >= 0) & (cls < len(self.names))).all():
                raise ValueError(f'Invalid classes/values in {label_path}')
            centers, sizes = targets[:, 1:3], targets[:, 3:5]
            if not ((sizes > 0).all() and (sizes <= 1).all() and
                    (centers-sizes/2 >= -1e-7).all() and (centers+sizes/2 <= 1+1e-7).all()):
                raise ValueError(f'Invalid normalized boxes in {label_path}')
        ignores = np.asarray(json.loads(ignore_path.read_text()), dtype=np.float32).reshape(-1, 4)
        with Image.open(image_path) as source:
            width, height = source.size
            if len(ignores) and not (np.isfinite(ignores).all() and
                    (ignores[:, :2] >= 0).all() and
                    (ignores[:, 2:] > ignores[:, :2]).all() and
                    (ignores[:, [0, 2]] <= width).all() and
                    (ignores[:, [1, 3]] <= height).all()):
                raise ValueError(f'Invalid absolute ignore boxes in {ignore_path}')
            augmentation_info = None
            if self.augment is not None:
                worker = torch.utils.data.get_worker_info()
                seed = worker.seed if worker is not None else self.seed
                if self._rng is None or self._rng_seed != seed:
                    self._rng = random.Random(seed)
                    self._rng_seed = seed
                source, targets, ignores, augmentation_info = self.augment(source, targets, ignores, self._rng)
            image, targets, ignores, meta = letterbox(source, targets, ignores, self.image_size)
        meta.update(augmented=augmentation_info is not None, augmentation=augmentation_info)
        meta.update(image_id=stem, image_path=str(image_path))
        return dict(image=torch.from_numpy(normalize_image(image)),
                    targets=torch.from_numpy(targets),
                    ignore_boxes=torch.from_numpy(ignores), metadata=meta)


def detection_collate(samples):
    """Batch variable object counts; targets are [batch_id,class_id,xc,yc,w,h]."""
    rows = []
    for batch_index, sample in enumerate(samples):
        targets = sample['targets']
        batch_ids = targets.new_full((len(targets), 1), batch_index)
        rows.append(torch.cat((batch_ids, targets), dim=1))
    return dict(images=torch.stack([s['image'] for s in samples]),
                targets=torch.cat(rows, dim=0),
                ignore_boxes=[s['ignore_boxes'] for s in samples],
                metadata=[s['metadata'] for s in samples])


def make_dataloader(config_path='configs/kitti.yaml', split='train',
                    split_dir='splits/kitti', batch_size=8, num_workers=0, augment=None):
    config = yaml.safe_load(Path(config_path).read_text())
    generator = torch.Generator().manual_seed(int(config.get('seed', 42)))
    dataset = KITTIDataset(config_path, split, split_dir, augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                      num_workers=num_workers, collate_fn=detection_collate,
                      generator=generator, pin_memory=torch.cuda.is_available(), drop_last=False)
