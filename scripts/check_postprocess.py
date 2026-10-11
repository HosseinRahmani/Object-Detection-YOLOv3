"""Deterministic unit checks for decoding, NMS and original-image geometry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils.postprocess import postprocess_predictions,class_aware_nms,restore_boxes


def main():
    torch.set_num_threads(2)
    anchors = torch.tensor([[[20,10]]*3,[[10,10]]*3,[[5,5]]*3],dtype=torch.float32)
    outputs = [torch.full((2,3,g,g,7),-100.) for g in (1,2,4)]
    outputs[0][0,0,0,0] = torch.tensor([0.,0.,0.,0.,10.,10.,-10.])
    results = postprocess_predictions(outputs,anchors,(64,64))
    assert results[0].shape == (1,6) and results[1].shape == (0,6)
    torch.testing.assert_close(results[0][0,:4],torch.tensor([22.,27.,42.,37.]))
    assert results[0][0,4] > .99 and results[0][0,5] == 0
    # Same-class duplicate is suppressed, different class remains.
    duplicates = torch.tensor([[1.,1.,20.,20.,.9,0],[1.,1.,20.,20.,.8,0],[1.,1.,20.,20.,.7,1]])
    kept = class_aware_nms(duplicates)
    assert len(kept) == 2 and kept[:,5].tolist() == [0.,1.]
    meta = dict(scale=(2.,2.),padding=(0,100),original_size=(100,200),input_size=(400,400))
    restored = restore_boxes(torch.tensor([[40.,120.,200.,200.],[-10.,50.,900.,500.]]),meta)
    torch.testing.assert_close(restored,torch.tensor([[20.,10.,100.,50.],[0.,0.,200.,100.]]))
    outputs[0][0,0,0,0,6] = 10
    multi = postprocess_predictions(outputs,anchors,(64,64),multi_label=True)
    assert len(multi[0]) == 2
    outputs[0][0,0,0,0,0] = float('nan')
    assert not len(postprocess_predictions(outputs,anchors,(64,64))[0])
    # A box entirely in letterbox padding is discarded after inverse clipping.
    outputs = [torch.full((1,3,g,g,7),-100.) for g in (1,2,4)]
    outputs[2][0,0,0,0] = torch.tensor([0.,0.,0.,0.,10.,10.,-10.])
    padmeta = dict(scale=(.32,.32),padding=(0,16),original_size=(100,200),input_size=(64,64))
    assert not len(postprocess_predictions(outputs,anchors,(64,64),[padmeta])[0])
    print('Passed: decoding, confidence, empty outputs, class-aware NMS, multi-label mode, NaN filtering, inverse coordinates and padding removal.')


if __name__ == '__main__':
    main()
