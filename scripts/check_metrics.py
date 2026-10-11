"""Known-answer checks for matching, AP, ignore regions and loader geometry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from utils.metrics import compute_metrics
from utils.transforms import letterbox
from utils.dataset import detection_collate
from evaluate import original_annotations, check_splits
import tempfile


def record(predictions, targets, ignores=(), image_id='one'):
    return dict(image_id=image_id, predictions=predictions, targets=targets, ignore_boxes=ignores)


def close(actual, expected):
    assert abs(actual-expected) < 1e-8, (actual, expected)


def main():
    box = [0, 0, 10, 10]
    gt = [[0, *box]]
    perfect = [[*box, .9, 0]]
    result, _ = compute_metrics([record(perfect, gt)], ['Car'])
    for key in ('precision', 'recall', 'f1', 'mAP50', 'mAP75', 'mAP50_95'):
        close(result['summary'][key], 1)
    # One-to-one duplicates: a second prediction is FP, never another TP.
    result, _ = compute_metrics([record(perfect+[[*box,.8,0]], gt)], ['Car'])
    close(result['summary']['precision'], .5)
    close(result['summary']['mAP50'], 1)  # FP after full recall doesn't reduce envelope AP.
    # High-score FP before TP gives precision envelope .5 at every recall point.
    result, _ = compute_metrics([record([[20,20,30,30,.99,0]]+perfect, gt)], ['Car'])
    close(result['summary']['mAP50'], .5)
    # mAP is a class mean, not weighted by the number of labeled objects.
    result, _ = compute_metrics([record(perfect+[[20,20,30,30,.8,0]],
        gt+[[0,20,20,30,30],[1,40,40,50,50]])], ['Car','Van'])
    close(result['summary']['mAP50'], .5)
    close(result['summary']['recall'], 2/3)
    # Partial recall: 51 of the 101 recall samples are covered (0 through .5).
    result, _ = compute_metrics([record(perfect, gt+[[0,20,20,30,30]])], ['Car'])
    close(result['summary']['recall'], .5)
    close(result['summary']['mAP50'], 51/101)
    # IoU .6 passes .50,.55,.60 only, so AP50:.95 is .3.
    result, _ = compute_metrics([record([[0,0,6,10,.9,0]], gt)], ['Car'])
    close(result['summary']['mAP50'], 1)
    close(result['summary']['mAP75'], 0)
    close(result['summary']['mAP50_95'], .3)
    # Wrong class: FP for predicted class, FN for target class. Absent class AP null.
    result, _ = compute_metrics([record([[*box,.9,1]], gt)], ['Car','Van'])
    assert result['summary']['fp'] == 1 and result['summary']['fn'] == 1
    assert result['per_class'][1]['AP50'] is None
    assert result['summary']['classes_in_mAP'] == ['Car']
    # DontCare uses prediction-area overlap, valid matches and duplicates take priority.
    result, _ = compute_metrics([record(perfect+[[*box,.8,0],[21,21,22,22,.7,0]], gt,
                                               [box,[20,20,30,30]])], ['Car'])
    assert result['summary']['tp'] == 1 and result['summary']['fp'] == 1
    assert result['summary']['ignored'] == 1
    # Empty outputs still count missed targets. Images with no GT retain false positives.
    result, _ = compute_metrics([record([], gt), record(perfect, [], image_id='two')], ['Car'])
    assert result['summary']['fn'] == 1 and result['summary']['fp'] == 1
    close(result['summary']['mAP50'], 0)
    result, _ = compute_metrics([record([], [])], ['Car'])
    assert result['summary']['mAP50'] is None and result['summary']['recall'] is None
    # Operating confidence changes P/R counts, not AP ranking input.
    result, _ = compute_metrics([record([[*box,.1,0]], gt)], ['Car'], operating_confidence=.25)
    close(result['summary']['mAP50'], 1)
    close(result['summary']['recall'], 0)
    # Confidence ordering must be global across images, not just per-image.
    result, _ = compute_metrics([record(perfect, gt),
                                record([[20,20,30,30,.99,0]], [], image_id='two')], ['Car'])
    close(result['summary']['mAP50'], .5)
    # Real letterbox -> collate -> inverse annotation geometry, including rounding.
    targets = np.array([[0,.5,.5,.4,.5]], dtype=np.float32)
    ignores = np.array([[10,5,30,15]], dtype=np.float32)
    rgb, labels, ignore, meta = letterbox(Image.new('RGB',(101,37)),targets,ignores,64)
    sample = dict(image=torch.zeros(3,64,64), targets=torch.from_numpy(labels),
                  ignore_boxes=torch.from_numpy(ignore), metadata=meta)
    original_gt, original_ignore = original_annotations(detection_collate([sample]), 0)
    np.testing.assert_allclose(original_gt, [[0,30.3,9.25,70.7,27.75]], atol=1e-4)
    np.testing.assert_allclose(original_ignore, ignores, atol=1e-4)
    # Reject NaNs, invalid boxes/classes, duplicate IDs and leaking split membership.
    invalid = [record([[*box,float('nan'),0]], gt), record([[0,0,0,10,.9,0]],gt),
               record([[*box,.9,2]],gt)]
    for bad in invalid:
        try:
            compute_metrics([bad], ['Car'])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid record accepted')
    try:
        compute_metrics([record([],[]), record([],[])], ['Car'])
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate image IDs accepted')
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for split, value in [('train','a'),('val','b'),('test','c')]:
            (root/f'{split}.txt').write_text(value+'\n')
        check_splits(root, dict(train_ids=['a'], val_ids=['b']))
        (root/'test.txt').write_text('a\n')
        try:
            check_splits(root, {})
        except ValueError:
            pass
        else:
            raise AssertionError('Split overlap accepted')
    print('Passed: known AP values, IoU thresholds, one-to-one matching, wrong classes,')
    print('DontCare priority/IoA, empty cases, confidence ordering, inverse GT geometry,')
    print('invalid inputs and split leakage checks.')


if __name__ == '__main__':
    main()
