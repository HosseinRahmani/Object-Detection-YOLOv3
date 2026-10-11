"""Predict boxes on validation samples or external images; save JSON and previews."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image,ImageDraw
from models.yolov3 import YOLOv3
from utils.dataset import make_dataloader
from utils.transforms import letterbox,normalize_image
from utils.postprocess import postprocess_predictions


def draw_predictions(source,detections,names,destination):
    with Image.open(source) as file:
        image = file.convert('RGB')
    draw = ImageDraw.Draw(image)
    colors = ['#ff5555','#55dd55','#5599ff','#ffcc00','#dd55ff','#00dddd','#ff8800','#aabbff']
    for x1,y1,x2,y2,confidence,class_id in detections.tolist():
        cls = int(class_id); color = colors[cls % len(colors)]
        draw.rectangle((x1,y1,x2,y2),outline=color,width=3)
        text = f'{names[cls]} {confidence:.2f}'
        # Anchor label at the box upper-right, within image boundaries.
        bounds = draw.textbbox((0,0),text)
        tw,th = bounds[2]-bounds[0],bounds[3]-bounds[1]
        x = max(0,min(image.width-tw-4,x2-tw-4))
        y = max(0,min(image.height-th-4,y1-th-6))
        draw.rectangle((x,y,x+tw+4,y+th+4),fill='black')
        draw.text((x+2,y+2-bounds[1]),text,fill=color)
    image.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=Path('outputs/finetune/best.pt'))
    parser.add_argument('--config',default='configs/kitti.yaml')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--source',type=Path,help='Image file or directory')
    source.add_argument('--split',choices=['train','val','test'],help='Default: val')
    parser.add_argument('--split-dir',default='splits/kitti')
    parser.add_argument('--limit',type=int,default=5,help='Images to preview; 0 means all')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--iou',type=float,default=.45)
    parser.add_argument('--max-detections',type=int,default=300)
    parser.add_argument('--multi-label',action='store_true')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--output',type=Path,default=Path('outputs/predictions'))
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error('Invalid limit/batch size')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output')
    checkpoint = torch.load(args.weights,map_location='cpu',weights_only=True)
    config = yaml.safe_load(Path(args.config).read_text())
    if checkpoint['names'] != config['names'] or checkpoint['image_size'] != int(config.get('image_size',416)):
        raise ValueError('Checkpoint class ordering/image size differs from config')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    model = YOLOv3(checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict']);model.to(device).eval()
    size,names = checkpoint['image_size'],checkpoint['names']
    checkpoint_epoch = checkpoint.get('epoch')
    checkpoint_epoch = checkpoint_epoch+1 if checkpoint_epoch is not None and checkpoint_epoch >= 0 else None
    del checkpoint  # Release optimizer and duplicate CPU weights from training checkpoint.
    if args.source:
        extensions = {'.png','.jpg','.jpeg','.bmp','.webp'}
        if args.source.is_dir():
            files = sorted(p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        elif args.source.is_file():
            files = [args.source]
        else:
            raise FileNotFoundError(args.source)
        if args.limit:files = files[:args.limit]
        if not files:raise ValueError('No images found')
        def batches():
            for start in range(0,len(files),args.batch_size):
                images,metadata = [],[]
                for path in files[start:start+args.batch_size]:
                    with Image.open(path) as source_image:
                        rgb,_,_,meta = letterbox(source_image,np.empty((0,5)),np.empty((0,4)),size)
                    images.append(torch.from_numpy(normalize_image(rgb)))
                    meta.update(image_path=str(path),image_id=path.stem,augmented=False)
                    metadata.append(meta)
                yield dict(images=torch.stack(images),metadata=metadata)
        batch_source = batches()
    else:
        batch_source = make_dataloader(args.config,args.split or 'val',args.split_dir,args.batch_size,augment=False)
    args.output.mkdir(parents=True,exist_ok=True)
    records,count = [],0
    with torch.inference_mode():
        for batch in batch_source:
            if args.limit and count >= args.limit:break
            images = batch['images'].to(device)
            detections = postprocess_predictions(model(images),model.anchors,images.shape[-2:],batch['metadata'],
                confidence_threshold=args.conf,iou_threshold=args.iou,max_detections=args.max_detections,
                multi_label=args.multi_label)
            for meta,prediction in zip(batch['metadata'],detections):
                if args.limit and count >= args.limit:break
                filename = f'{count:06d}_{meta["image_id"]}.png'
                draw_predictions(meta['image_path'],prediction,names,args.output/filename)
                records.append(dict(image_id=meta['image_id'],source=meta['image_path'],preview=filename,
                    detections=[dict(box_xyxy=row[:4],confidence=row[4],class_id=int(row[5]),class_name=names[int(row[5])])
                                for row in prediction.tolist()]))
                count += 1
    (args.output/'predictions.json').write_text(json.dumps(dict(
        weights=str(args.weights),checkpoint_epoch=checkpoint_epoch,
        coordinate_system='original image pixels, xyxy',confidence_threshold=args.conf,
        nms_iou=args.iou,multi_label=args.multi_label,max_detections=args.max_detections,
        names=names,images=records),indent=2))
    print(f'Device: {device}; saved {count} images and predictions.json to {args.output}')
    print('Displayed scores are confidence, not accuracy. AP/mAP evaluation is the next step.')


if __name__ == '__main__':
    main()
