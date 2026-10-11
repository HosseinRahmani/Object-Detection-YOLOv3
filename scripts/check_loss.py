"""Check loss behavior and gradients with synthetic raw predictions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.yolov3 import YOLOv3
from models.loss import YOLOv3Loss


def main():
    torch.set_num_threads(2)
    anchors = torch.tensor([[[116,90],[156,198],[373,326]],
                            [[30,61],[62,45],[59,119]],[[10,13],[16,30],[33,23]]],dtype=torch.float32)
    criterion = YOLOv3Loss(anchors,8)
    def outputs():
        return [torch.zeros(2,3,g,g,13,requires_grad=True) for g in (2,4,8)]
    targets = torch.tensor([[0,0,.5,.5,.2,.3],[1,5,.3,.4,.1,.2]],dtype=torch.float32)
    predictions = outputs()
    loss,stats = criterion(predictions,targets,[torch.empty(0,4),torch.tensor([[.7,.7,.9,.9]])],(64,64))
    loss.backward()
    assert stats['positives'] == 2 and stats['unassigned_targets'] == 0
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictions)
    empty_predictions = outputs()
    empty_loss,_ = criterion(empty_predictions,torch.empty(0,6),[torch.empty(0,4)]*2,(64,64))
    empty_loss.backward()
    assert torch.isfinite(empty_loss)
    ignored_predictions = outputs()
    ignored_loss,_ = criterion(ignored_predictions,torch.empty(0,6),[torch.tensor([[-100.,-100.,100.,100.]])]*2,(64,64))
    # A region covering all predictions must eliminate all background penalties.
    assert ignored_loss.item() == 0
    ignored_loss.backward()
    # Also verify actual model forward output geometry.
    model = YOLOv3(8).eval()
    with torch.no_grad():
        actual = model(torch.zeros(1,3,64,64))
    assert [tuple(p.shape) for p in actual] == [(1,3,2,2,13),(1,3,4,4,13),(1,3,8,8,13)]
    print('Passed: positive assignment, finite loss/gradients, empty targets, ignore masks and model forward.')
    print(stats)


if __name__ == '__main__':
    main()
