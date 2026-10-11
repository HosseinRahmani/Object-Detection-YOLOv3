"""Evaluate this project's YOLOv3 checkpoint on labeled KITTI val/test splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml

from models.yolov3 import YOLOv3
from utils.dataset import make_dataloader
from utils.postprocess import postprocess_predictions, restore_boxes
from utils.metrics import compute_metrics


def original_annotations(batch, index):
    """Convert loader targets/ignore regions back to original image pixels."""
    meta = batch['metadata'][index]
    height, width = meta['input_size']
    target = batch['targets'][batch['targets'][:, 0] == index, 1:]
    center, size = target[:, 1:3], target[:, 3:5]
    boxes = torch.cat((center-size/2, center+size/2), 1)
    scale = boxes.new_tensor([width, height, width, height])
    boxes = restore_boxes(boxes*scale, meta)
    gt = torch.cat((target[:, :1], boxes), 1)
    ignore = restore_boxes(batch['ignore_boxes'][index]*scale, meta)
    return gt.numpy(), ignore.numpy()


def check_splits(directory, checkpoint):
    splits = {}
    for split in ('train', 'val', 'test'):
        ids = [line.strip() for line in (directory/f'{split}.txt').read_text().splitlines() if line.strip()]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f'Empty or duplicate IDs in {split}.txt')
        splits[split] = ids
    for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(splits[first]) & set(splits[second]):
            raise ValueError(f'{first}/{second} splits overlap')
    for split in ('train', 'val'):
        previous = checkpoint.get(f'{split}_ids')
        if previous is not None and set(previous) != set(splits[split]):
            raise ValueError(f'{split} split differs from saved training checkpoint')
    return splits


def write_csv(path, rows, fields):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('outputs/finetune/best.pt'))
    parser.add_argument('--config', default='configs/kitti.yaml')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--split-dir', type=Path, default=Path('splits/kitti'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--conf', type=float, default=.001, help='Low prediction floor for AP curves')
    parser.add_argument('--report-conf', type=float, default=.25, help='Confidence for P/R/F1 at IoU .50')
    parser.add_argument('--iou', type=float, default=.45, help='NMS IoU, separate from metric IoUs')
    parser.add_argument('--ignore-ioa', type=float, default=.5)
    parser.add_argument('--max-detections', type=int, default=300)
    parser.add_argument('--multi-label', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='0=full split; positive=subset smoke check')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, help='Default: outputs/metrics_<split>')
    args = parser.parse_args()
    if not 0 <= args.conf <= args.report_conf <= 1:
        parser.error('Require 0 <= --conf <= --report-conf <= 1')
    if not 0 <= args.iou <= 1 or not 0 <= args.ignore_ioa <= 1:
        parser.error('IoU/IoA thresholds must lie in [0,1]')
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0 or not 1 <= args.max_detections <= 10000:
        parser.error('Invalid batch size, workers, limit or detection cap')
    output = args.output or Path(f'outputs/metrics_{args.split}')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output')
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    config = yaml.safe_load(Path(args.config).read_text())
    if checkpoint['names'] != config['names'] or checkpoint['num_classes'] != len(config['names']) or checkpoint['image_size'] != int(config.get('image_size', 416)):
        raise ValueError('Checkpoint classes/image size differ from config')
    splits = check_splits(args.split_dir, checkpoint)
    provenance_verified = checkpoint.get('train_ids') is not None and checkpoint.get('val_ids') is not None
    device = torch.device(('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = YOLOv3(checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict']); model.to(device).eval()
    names = checkpoint['names']
    epoch = checkpoint.get('epoch')
    epoch = epoch+1 if epoch is not None and epoch >= 0 else None
    del checkpoint
    loader = make_dataloader(args.config, args.split, str(args.split_dir), args.batch_size,
                             args.num_workers, augment=False)
    records = []
    print(f'Device: {device}; {args.split}={len(loader.dataset)}; augmentation disabled', flush=True)
    with torch.inference_mode():
        for batch in loader:
            remaining = args.limit-len(records) if args.limit else len(batch['metadata'])
            if remaining <= 0:
                break
            images = batch['images'][:remaining].to(device)
            metadata = batch['metadata'][:len(images)]
            predictions = postprocess_predictions(model(images), model.anchors, images.shape[-2:], metadata,
                confidence_threshold=args.conf, iou_threshold=args.iou,
                max_detections=args.max_detections, multi_label=args.multi_label)
            for index, (meta, prediction) in enumerate(zip(metadata, predictions)):
                gt, ignore = original_annotations(batch, index)
                records.append(dict(image_id=meta['image_id'], predictions=prediction.numpy(),
                                    targets=gt, ignore_boxes=ignore))
            if len(records) % (args.batch_size*25) == 0:
                print(f'Evaluated {len(records)} images', flush=True)
    print(f'Computing ranked metrics for {len(records)} images...', flush=True)
    metrics, curves = compute_metrics(records, names, args.report_conf, args.ignore_ioa)
    partial = len(records) < len(loader.dataset)
    metrics['run'] = dict(protocol='project_all_classes_101_point_AP_v1', split=args.split,
        partial=partial, weights=str(args.weights), weights_sha256=file_hash(args.weights),
        checkpoint_epoch=epoch, checkpoint_split_provenance_verified=provenance_verified,
        names=names, config=config, device=str(device), torch_version=str(torch.__version__),
        evaluated_ids=[r['image_id'] for r in records], split_ids=splits,
        confidence_floor=args.conf, report_confidence=args.report_conf, report_match_iou=.5,
        nms_iou=args.iou, ignore_prediction_ioa=args.ignore_ioa,
        metric_ious=[round(.5+.05*i, 2) for i in range(10)], recall_samples=101,
        max_detections=args.max_detections, max_candidates=10000, multi_label=args.multi_label,
        coordinate_system='original image pixels, continuous xyxy',
        note='All sizes/difficulties; not official KITTI R40 or full COCO evaluation')
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    write_csv(output/'per_class.csv', metrics['per_class'],
              ['class_id','class_name','ground_truth','predictions','tp','fp','fn','ignored',
               'precision','recall','f1','AP50','AP75','AP50_95'])
    write_csv(output/'pr_curves.csv', curves, ['class_id','class_name','confidence','precision','recall','tp','fp'])
    serialized = [dict(image_id=r['image_id'], **{k:r[k].tolist() for k in ('predictions','targets','ignore_boxes')})
                  for r in records]
    (output/'predictions.json').write_text(json.dumps(dict(names=names, coordinate_system='original image pixels, xyxy',
                                                       images=serialized), allow_nan=False))
    if partial:
        print('SUBSET CHECK: these are not full-split results.')
    if not provenance_verified:
        print('Checkpoint lacks training split IDs; only current split disjointness was verified.')
    summary = metrics['summary']
    for key in ('precision','recall','f1','mAP50','mAP75','mAP50_95'):
        value = summary[key]
        print(f'{key}: {value:.4f}' if value is not None else f'{key}: undefined (no ground truth)')
    print(f'Saved metrics, class table, PR curves and predictions: {output}')


if __name__ == '__main__':
    main()
