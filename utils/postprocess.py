"""Decode our YOLOv3 outputs, filter confidence and apply class-aware NMS.

Returns CPU float32 tensors [N,6]: x1,y1,x2,y2,confidence,class_id.
Coordinates are absolute original-image pixels when metadata is supplied;
otherwise they are absolute model-input pixels. No GT/DontCare filtering here.
"""
import math
import torch


def box_iou_one(box,others):
    intersection = (torch.minimum(box[2:],others[:,2:])-
                    torch.maximum(box[:2],others[:,:2])).clamp_min(0).prod(-1)
    area = (box[2:]-box[:2]).clamp_min(0).prod()
    other_area = (others[:,2:]-others[:,:2]).clamp_min(0).prod(-1)
    return intersection/(area+other_area-intersection).clamp_min(1e-9)


def class_aware_nms(detections,iou_threshold=.45,max_detections=300):
    """CPU greedy NMS; only overlapping predictions of the same class compete."""
    if not 0 <= iou_threshold <= 1 or max_detections < 1:
        raise ValueError('Invalid NMS settings')
    detections = detections.detach().float().cpu()
    if not len(detections):
        return detections.reshape(0,6)
    order = detections[:,4].argsort(descending=True,stable=True)
    selected = []
    while len(order) and len(selected) < max_detections:
        index = order[0]
        selected.append(index)
        remaining = order[1:]
        if not len(remaining):
            break
        same_class = detections[remaining,5] == detections[index,5]
        ious = box_iou_one(detections[index,:4],detections[remaining,:4])
        order = remaining[~(same_class & (ious > iou_threshold))]
    return detections[torch.stack(selected)]


def restore_boxes(boxes,metadata):
    """Undo our deterministic letterbox resize/padding, then clip to source image."""
    if metadata.get('augmented',False):
        raise ValueError('Original-coordinate inference requires augmentation disabled')
    sx,sy = metadata['scale']
    left,top = metadata['padding']
    height,width = metadata['original_size']
    if sx <= 0 or sy <= 0:
        raise ValueError('Invalid resize scale')
    boxes = boxes.clone()
    boxes[:,[0,2]] = ((boxes[:,[0,2]]-left)/sx).clamp(0,width)
    boxes[:,[1,3]] = ((boxes[:,[1,3]]-top)/sy).clamp(0,height)
    return boxes


@torch.no_grad()
def postprocess_predictions(predictions,anchors,image_size,metadata=None,
                            confidence_threshold=.25,iou_threshold=.45,
                            max_detections=300,max_candidates=10000,multi_label=False):
    """Decode raw [B,3,grid_h,grid_w,5+C] outputs in stride 32,16,8 order.

    centers=(sigmoid(tx,ty)+cell)/grid_size
    sizes=exp(tw,th)*anchor; wh logits are clamped to [-10,10] for stability.
    confidence=sigmoid(objectness)*sigmoid(class_logit).
    Best-class mode is default; multi_label emits every class above threshold.
    Padding-only/invalid boxes are removed before NMS. Ignore regions are kept
    for later evaluation matching, never used to censor inference predictions.
    """
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence threshold must be between 0 and 1')
    if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError('NMS IoU threshold must be between 0 and 1')
    if max_detections < 1 or max_candidates < max_detections:
        raise ValueError('Require max_candidates >= max_detections >= 1')
    if len(predictions) != 3:
        raise ValueError('Expected three YOLOv3 output scales')
    batch = predictions[0].shape[0]
    classes = predictions[0].shape[-1]-5
    height,width = map(int,image_size)
    if classes < 1 or height <= 0 or width <= 0:
        raise ValueError('Invalid class count or image size')
    if metadata is not None and len(metadata) != batch:
        raise ValueError('One metadata record is required per image')
    anchors = torch.as_tensor(anchors,dtype=torch.float32,device=predictions[0].device)
    if anchors.shape != (3,3,2) or not torch.isfinite(anchors).all() or not (anchors > 0).all():
        raise ValueError('Expected positive finite anchors [3,3,2]')
    decoded,probabilities,valid_rows = [],[],[]
    for scale,raw in enumerate(predictions):
        if raw.ndim != 5 or raw.shape[0] != batch or raw.shape[1] != 3 or raw.shape[-1] != 5+classes:
            raise ValueError('Prediction shape mismatch')
        raw = raw.float()
        gh,gw = raw.shape[2:4]
        yy,xx = torch.meshgrid(torch.arange(gh,device=raw.device),torch.arange(gw,device=raw.device),indexing='ij')
        grid = torch.stack((xx,yy),dim=-1).float()
        centers = (raw[..., :2].sigmoid()+grid)/raw.new_tensor([gw,gh])*raw.new_tensor([width,height])
        sizes = raw[..., 2:4].clamp(-10,10).exp()*anchors[scale,:,None,None,:]
        boxes = torch.cat((centers-sizes/2,centers+sizes/2),dim=-1)
        scores = raw[...,4:5].sigmoid()*raw[...,5:].sigmoid()
        decoded.append(boxes.reshape(batch,-1,4))
        probabilities.append(scores.reshape(batch,-1,classes))
        valid_rows.append(torch.isfinite(raw).all(dim=-1).reshape(batch,-1))
    decoded = torch.cat(decoded,dim=1)
    probabilities = torch.cat(probabilities,dim=1)
    valid_rows = torch.cat(valid_rows,dim=1)
    results = []
    for b in range(batch):
        boxes,scores = decoded[b],probabilities[b]
        if multi_label:
            rows,class_ids = torch.where((scores >= confidence_threshold) & valid_rows[b,:,None])
            confidence = scores[rows,class_ids]
            boxes = boxes[rows]
        else:
            confidence,class_ids = scores.max(dim=-1)
            keep = (confidence >= confidence_threshold) & valid_rows[b]
            boxes,confidence,class_ids = boxes[keep],confidence[keep],class_ids[keep]
        # NMS on CPU is portable across CPU, CUDA and Apple MPS without torchvision.
        boxes,confidence,class_ids = boxes.cpu(),confidence.cpu(),class_ids.cpu()
        if metadata is not None:
            meta = metadata[b]
            if tuple(meta['input_size']) != (height,width):
                raise ValueError('Metadata input_size differs from decoded image size')
            boxes = restore_boxes(boxes,meta)
        else:
            boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,width)
            boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,height)
        keep = torch.isfinite(boxes).all(-1) & ((boxes[:,2:]-boxes[:,:2]) > 0).all(-1)
        detections = torch.cat((boxes[keep],confidence[keep,None],class_ids[keep,None].float()),dim=1)
        if len(detections) > max_candidates:
            detections = detections[detections[:,4].argsort(descending=True,stable=True)[:max_candidates]]
        results.append(class_aware_nms(detections,iou_threshold,max_detections))
    return results
