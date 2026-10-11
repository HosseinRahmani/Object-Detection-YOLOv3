"""Download original YOLOv3 COCO weights and save KITTI initialization checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from models.yolov3 import YOLOv3

WEIGHTS_URL = 'https://data.pjreddie.com/files/yolov3.weights'


def download_weights(destination):
    if destination.exists():
        print(f'Using existing weights: {destination}')
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.part')
    print('Downloading original COCO YOLOv3 weights (approximately 237 MiB)...',flush=True)
    try:
        request = urllib.request.Request(WEIGHTS_URL,headers={'User-Agent':'YOLOv3-KITTI-pipeline'})
        with urllib.request.urlopen(request,timeout=120) as source,temporary.open('wb') as output:
            while True:
                chunk = source.read(1024*1024)
                if not chunk:
                    break
                output.write(chunk)
        # Full standard weights have 62,001,757 float32 values + a 20-byte header.
        if temporary.stat().st_size != 248007048:
            raise ValueError('Download size does not match standard original yolov3.weights')
        os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/kitti.yaml')
    parser.add_argument('--weights',type=Path,default=Path('weights/yolov3.weights'))
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('weights/yolov3_kitti_init.pt'))
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error('Output checkpoint exists; choose another --output or use --overwrite')
    config = yaml.safe_load(Path(args.config).read_text())
    names = config['names']
    if not names or len(set(names)) != len(names) or 'DontCare' in names:
        raise ValueError('Expected unique target class names excluding DontCare')
    size = int(config.get('image_size',416))
    if size <= 0 or size % 32:
        raise ValueError('image_size must be a positive multiple of 32')
    if args.download:
        download_weights(args.weights)
    if not args.weights.is_file():
        parser.error(f'Weights missing: {args.weights}; add --download or download manually')
    torch.manual_seed(int(config.get('seed',42)))
    torch.set_num_threads(min(4,torch.get_num_threads()))
    model = YOLOv3(num_classes=len(names))
    report = model.load_darknet_weights(args.weights,source_classes=80)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.zeros(1,3,size,size))
    expected = [(1,3,size//s,size//s,5+len(names)) for s in (32,16,8)]
    shapes = [tuple(p.shape) for p in predictions]
    if shapes != expected or any(not torch.isfinite(p).all().item() for p in predictions):
        raise RuntimeError('Forward-pass shape/finiteness check failed')
    digest = hashlib.sha256()
    with args.weights.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):
            digest.update(chunk)
    checkpoint = dict(model_state_dict=model.state_dict(),num_classes=len(names),names=names,
                      image_size=size,source_url=WEIGHTS_URL,source_sha256=digest.hexdigest(),
                      load_report=report,stage='pretrained_initialization',epoch=-1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,args.output)
    print(json.dumps(report,indent=2))
    print('Raw output shapes:',shapes)
    print(f'Saved initialization checkpoint: {args.output}')
    print('KITTI output layers are newly initialized; fine-tuning is still required.')


if __name__ == '__main__':
    main()
