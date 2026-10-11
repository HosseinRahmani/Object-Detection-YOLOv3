"""Smoke-check a batch and save boxes after preprocessing. Run from project root."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
from utils.dataset import make_dataloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/kitti.yaml')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--split-dir', default='splits/kitti')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('outputs/preprocessing_checks'))
    args = parser.parse_args()
    loader = make_dataloader(args.config, args.split, args.split_dir, args.batch_size)
    batch = next(iter(loader))
    print('Images:', tuple(batch['images'].shape), batch['images'].dtype)
    print('Pixel range:', batch['images'].min().item(), batch['images'].max().item())
    print('Targets:', tuple(batch['targets'].shape))
    print('Ignore boxes per image:', [len(x) for x in batch['ignore_boxes']])
    args.output.mkdir(parents=True, exist_ok=True)
    for i, tensor in enumerate(batch['images']):
        rgb = (tensor.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        targets = batch['targets'][batch['targets'][:, 0] == i, 1:].numpy()
        for cls, xc, yc, bw, bh in targets:
            box = ((xc-bw/2)*width, (yc-bh/2)*height,
                   (xc+bw/2)*width, (yc+bh/2)*height)
            draw.rectangle(box, outline='lime', width=2)
            draw.text((max(0, box[0]), max(0, box[1]-12)), loader.dataset.names[int(cls)],
                      fill='lime', stroke_width=1, stroke_fill='black')
        for box in batch['ignore_boxes'][i].numpy():
            xyxy = box*np.array([width, height, width, height])
            draw.rectangle(tuple(xyxy), outline='white', width=2)
            draw.text((xyxy[0], xyxy[1]), 'DontCare', fill='white', stroke_width=1, stroke_fill='black')
        image.save(args.output/f'{args.split}_{batch["metadata"][i]["image_id"]}.png')
    print('Saved previews to', args.output)


if __name__ == '__main__':
    main()
