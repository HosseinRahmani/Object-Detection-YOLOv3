"""Standalone 2D detection metrics; original pixels, continuous xyxy boxes.

Project protocol: all configured classes, all object sizes, class-aware greedy
one-to-one matching, IoUs .50:.05:.95, 101 recall samples. This is neither the
official KITTI R40 evaluator nor the full COCO evaluator.
"""
import numpy as np


IOU_THRESHOLDS = np.round(np.linspace(.50, .95, 10), 2)


def overlaps(box, boxes, ioa=False):
    """IoU, or intersection divided by prediction area for DontCare regions."""
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) -
                              np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    other = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area if ioa else area + other - intersection, 1e-12)


def average_precision(recall, precision):
    """Mean of the precision envelope at 101 recall points (0 through 1)."""
    if not len(recall):
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recall, np.linspace(0, 1, 101), side='left')
    values = np.zeros(101)
    valid = positions < len(envelope)
    values[valid] = envelope[positions[valid]]
    return float(values.mean())


def _array(value, columns, label):
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, columns)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f'{label}: expected finite [N,{columns}] array')
    return array


def _check_classes(classes, count):
    if not ((classes == np.floor(classes)) & (classes >= 0) & (classes < count)).all():
        raise ValueError('Invalid class IDs')


def compute_metrics(records, names, operating_confidence=.25, ignore_ioa=.5):
    """records: image_id, predictions [xyxy,score,class], targets [class,xyxy],
    ignore_boxes [xyxy]. DontCare removes only otherwise unmatched predictions;
    valid target matching takes priority. Duplicates of valid targets remain FP.
    AP ranks all supplied predictions, independently of operating_confidence.
    Classes without GT have AP/recall null and are excluded from mean AP; their
    false positives are still included in overall (micro) precision.
    """
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected unique class names')
    if not 0 <= operating_confidence <= 1 or not 0 <= ignore_ioa <= 1:
        raise ValueError('Invalid confidence/ignore threshold')
    samples, seen = [], set()
    for record in records:
        image_id = record['image_id']
        if image_id in seen:
            raise ValueError(f'Duplicate image ID: {image_id}')
        seen.add(image_id)
        pred = _array(record['predictions'], 6, 'predictions')
        gt = _array(record['targets'], 5, 'targets')
        ignore = _array(record.get('ignore_boxes', []), 4, 'ignore_boxes')
        _check_classes(pred[:, 5], len(names)); _check_classes(gt[:, 0], len(names))
        if not ((pred[:, 4] >= 0) & (pred[:, 4] <= 1)).all():
            raise ValueError('Confidence must lie in [0,1]')
        for boxes in (pred[:, :4], gt[:, 1:], ignore):
            if not (boxes[:, 2:] > boxes[:, :2]).all():
                raise ValueError('Box widths/heights must be positive')
        samples.append((pred, gt, ignore))
    if not samples:
        raise ValueError('No images to evaluate')
    per_class, curves = [], []
    for cls, name in enumerate(names):
        gts = [gt[gt[:, 0] == cls, 1:] for _, gt, _ in samples]
        n_gt = sum(len(gt) for gt in gts)
        candidates = [(float(row[4]), image, row[:4])
                      for image, (pred, _, _) in enumerate(samples)
                      for row in pred[pred[:, 5] == cls]]
        candidates.sort(key=lambda item: -item[0])  # Stable ties: image/row order.
        scores = np.array([item[0] for item in candidates])
        aps, operating = {}, None
        for threshold in IOU_THRESHOLDS:
            used = [np.zeros(len(gt), dtype=bool) for gt in gts]
            flags = []  # 1=TP, 0=FP, -1=ignored; confidence sorted.
            for _, image, box in candidates:
                ious = overlaps(box, gts[image])
                available = np.where((ious >= threshold) & ~used[image])[0]
                if len(available):
                    match = available[np.argmax(ious[available])]
                    used[image][match] = True
                    flags.append(1)
                elif (ious >= threshold).any():
                    flags.append(0)  # Duplicate, even if a DontCare overlaps.
                elif (overlaps(box, samples[image][2], ioa=True) >= ignore_ioa).any():
                    flags.append(-1)
                else:
                    flags.append(0)
            flags = np.asarray(flags, dtype=int)
            valid = flags >= 0
            tp = np.cumsum(flags[valid] == 1)
            fp = np.cumsum(flags[valid] == 0)
            recall = tp / n_gt if n_gt else np.zeros(len(tp))
            precision = tp / np.maximum(tp + fp, 1)
            aps[f'{threshold:.2f}'] = average_precision(recall, precision) if n_gt else None
            if threshold == .5:
                selected = scores >= operating_confidence
                ntp = int(((flags == 1) & selected).sum())
                nfp = int(((flags == 0) & selected).sum())
                ignored = int(((flags == -1) & selected).sum())
                p = ntp / (ntp + nfp) if ntp + nfp else 0.0
                r = ntp / n_gt if n_gt else None
                operating = dict(tp=ntp, fp=nfp, fn=n_gt-ntp, ignored=ignored,
                                 precision=p, recall=r,
                                 f1=2*p*r/(p+r) if r is not None and p+r else 0.0)
                for score, p_value, r_value, nt, nf in zip(scores[valid], precision, recall, tp, fp):
                    curves.append(dict(class_id=cls, class_name=name, confidence=float(score),
                                       precision=float(p_value), recall=float(r_value) if n_gt else None,
                                       tp=int(nt), fp=int(nf)))
        per_class.append(dict(class_id=cls, class_name=name, ground_truth=n_gt,
                              predictions=len(candidates), AP50=aps['0.50'], AP75=aps['0.75'],
                              AP50_95=float(np.mean(list(aps.values()))) if n_gt else None,
                              AP_by_IoU=aps, **operating))
    tp = sum(row['tp'] for row in per_class)
    fp = sum(row['fp'] for row in per_class)
    fn = sum(row['fn'] for row in per_class)
    p = tp / (tp+fp) if tp+fp else 0.0
    r = tp / (tp+fn) if tp+fn else None
    present = [row for row in per_class if row['ground_truth']]
    summary = dict(images=len(samples), ground_truth=tp+fn, tp=tp, fp=fp, fn=fn,
                   ignored=sum(row['ignored'] for row in per_class), precision=p, recall=r,
                   f1=2*p*r/(p+r) if r is not None and p+r else 0.0,
                   classes_in_mAP=[row['class_name'] for row in present],
                   **{key: float(np.mean([row[source] for row in present])) if present else None
                      for key, source in [('mAP50', 'AP50'), ('mAP75', 'AP75'), ('mAP50_95', 'AP50_95')]})
    return dict(summary=summary, per_class=per_class), curves
