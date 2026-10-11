"""Fine-tune standalone YOLOv3 on prepared KITTI splits."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml
from models.yolov3 import YOLOv3
from models.loss import YOLOv3Loss
from utils.dataset import make_dataloader


def save_checkpoint(checkpoint,path):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint,temporary)
    os.replace(temporary,path)


def run_epoch(model,criterion,loader,device,optimizer=None,freeze_backbone=False,max_batches=None):
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        model.backbone.eval()  # Do not change frozen batch-normalization statistics.
    totals = {}; samples = 0
    with torch.set_grad_enabled(training):
        for index,batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            ignores = [b.to(device) for b in batch['ignore_boxes']]
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss,stats = criterion(predictions,targets,ignores,images.shape[-2:])
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite loss at batch {index}; image IDs: {[m["image_id"] for m in batch["metadata"]]}')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10.,error_if_nonfinite=True)
                optimizer.step()
            count = len(images); samples += count
            for key,value in stats.items():
                totals[key] = totals.get(key,0)+value*(1 if key in ('positives','unassigned_targets') else count)
            if (index+1) % 50 == 0:
                print(f'  {"train" if training else "val"} batch {index+1}: loss={stats["loss"]:.4f}',flush=True)
    if not samples:
        raise ValueError('No batches available')
    return {k:v if k in ('positives','unassigned_targets') else v/samples for k,v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/kitti.yaml')
    parser.add_argument('--weights',type=Path,default=Path('weights/yolov3_kitti_init.pt'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--split-dir',default='splits/kitti')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--freeze-epochs',type=int,default=3)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda, or mps')
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/finetune'))
    parser.add_argument('--max-batches',type=int,help='Limit train/val batches for smoke checks only')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.freeze_epochs < 0 or args.patience < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Invalid training settings')
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and args.resume is None:
        parser.error('Run directory is not empty; choose a new --run-dir or --resume last.pt')
    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config.get('seed',42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    init = torch.load(args.resume or args.weights,map_location='cpu',weights_only=True)
    if init['names'] != config['names'] or init['image_size'] != int(config.get('image_size',416)):
        raise ValueError('Checkpoint class ordering/image size differs from configuration')
    model = YOLOv3(len(config['names']))
    model.load_state_dict(init['model_state_dict'])
    model.to(device)
    criterion = YOLOv3Loss(model.anchors,model.num_classes).to(device)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {'params':list(model.backbone.parameters()),'lr':args.lr*.1},
        {'params':[p for p in model.parameters() if id(p) not in backbone_ids],'lr':args.lr}
    ],weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.01)
    train_loader = make_dataloader(args.config,'train',args.split_dir,args.batch_size,args.workers)
    val_loader = make_dataloader(args.config,'val',args.split_dir,args.batch_size,args.workers,augment=False)
    train_ids,val_ids = set(train_loader.dataset.ids),set(val_loader.dataset.ids)
    if train_ids & val_ids:
        raise ValueError('Train/validation split overlap')
    start_epoch,best,stale = 0,float('inf'),0
    history = args.run_dir/'history.csv'
    if args.resume:
        if init.get('stage') != 'fine_tuning':
            raise ValueError('--resume requires a fine-tuning checkpoint')
        previous_args = init['training_args']
        for key in ('epochs','batch_size','workers','lr','freeze_epochs','max_batches'):
            if previous_args[key] != getattr(args,key):
                raise ValueError(f'Resume must keep {key}={previous_args[key]}')
        if init['config'] != config or init['train_ids'] != train_loader.dataset.ids or init['val_ids'] != val_loader.dataset.ids:
            raise ValueError('Resume config/splits differ from checkpoint')
        optimizer.load_state_dict(init['optimizer_state_dict'])
        scheduler.load_state_dict(init['scheduler_state_dict'])
        start_epoch,best,stale = init['epoch']+1,init['best_val_loss'],init['stale_epochs']
        train_loader.generator.set_state(init['loader_generator_state'])
        if init.get('augmentation_rng_state') is not None:
            train_loader.dataset._rng = random.Random()
            train_loader.dataset._rng.setstate(init['augmentation_rng_state'])
            train_loader.dataset._rng_seed = seed
        torch.set_rng_state(init['torch_rng_state'])
        if device.type == 'cuda' and init.get('cuda_rng_states') is not None:
            torch.cuda.set_rng_state_all(init['cuda_rng_states'])
        if history.exists():
            with history.open() as file:
                rows = list(csv.DictReader(file))
            if rows and int(rows[-1]['epoch']) != start_epoch:
                raise ValueError('History differs from resume checkpoint; resume last.pt or use a new run directory')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    print(f'Device: {device}; train={len(train_ids)}, val={len(val_ids)}',flush=True)
    print('Best checkpoint selected by validation LOSS; mAP evaluation is a later step.',flush=True)
    if args.max_batches:
        print('SMOKE RUN: limited batches; these are not full-dataset results.',flush=True)
    for epoch in range(start_epoch,args.epochs):
        frozen = epoch < args.freeze_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(not frozen)
        started = time.time()
        used_lr = optimizer.param_groups[1]['lr']
        train_stats = run_epoch(model,criterion,train_loader,device,optimizer,frozen,args.max_batches)
        val_stats = run_epoch(model,criterion,val_loader,device,max_batches=args.max_batches)
        scheduler.step()
        improved = val_stats['loss'] < best
        if improved:
            best,stale = val_stats['loss'],0
        else:
            stale += 1
        row = dict(epoch=epoch+1,lr=used_lr,backbone_frozen=frozen,
                   seconds=round(time.time()-started,2))
        row.update({f'train_{k}':v for k,v in train_stats.items()})
        row.update({f'val_{k}':v for k,v in val_stats.items()})
        write_header = not history.exists()
        with history.open('a',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=list(row))
            if write_header:writer.writeheader()
            writer.writerow(row)
        rng = train_loader.dataset._rng
        checkpoint = dict(model_state_dict=model.state_dict(),names=config['names'],num_classes=model.num_classes,
                          image_size=int(config.get('image_size',416)),stage='fine_tuning',epoch=epoch,
                          optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                          best_val_loss=best,stale_epochs=stale,config=config,
                          training_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                          train_ids=train_loader.dataset.ids,val_ids=val_loader.dataset.ids,
                          loader_generator_state=train_loader.generator.get_state(),
                          augmentation_rng_state=rng.getstate() if rng else None,
                          torch_rng_state=torch.get_rng_state(),
                          cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                          source_sha256=init.get('source_sha256'),validation_stats=val_stats)
        save_checkpoint(checkpoint,args.run_dir/'last.pt')
        if improved:save_checkpoint(checkpoint,args.run_dir/'best.pt')
        print(f'Epoch {epoch+1}/{args.epochs}: train={train_stats["loss"]:.4f}, val={val_stats["loss"]:.4f}, frozen={frozen}',flush=True)
        if train_stats['unassigned_targets'] or val_stats['unassigned_targets']:
            print('Warning: assignment collisions left some targets unassigned; counts are in history.csv',flush=True)
        if stale >= args.patience:
            print('Early stopping: validation loss stopped improving.',flush=True)
            break
    print(f'Checkpoints and history: {args.run_dir}',flush=True)


if __name__ == '__main__':
    main()
