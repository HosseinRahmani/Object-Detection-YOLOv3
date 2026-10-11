"""YOLOv3-style loss for our raw three-scale output and KITTI loader.

Single-anchor assignment with collision fallback, BCE xy/objectness/classes,
MSE log-width/height, plus ignored negatives near GT or in DontCare regions.
This documented loss variant is not a bit-for-bit Darknet loss reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F


def xywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2]-boxes[..., 2:]/2,
                      boxes[..., :2]+boxes[..., 2:]/2),dim=-1)


def pairwise_overlap(boxes, references, ioa=False):
    low = torch.maximum(boxes[:,None,:2],references[None,:,:2])
    high = torch.minimum(boxes[:,None,2:],references[None,:,2:])
    intersection = (high-low).clamp_min(0).prod(dim=-1)
    area = (boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(dim=-1)
    ref_area = (references[:,2:]-references[:,:2]).clamp_min(0).prod(dim=-1)
    denominator = area[:,None] if ioa else area[:,None]+ref_area[None,:]-intersection
    return intersection/denominator.clamp_min(1e-9)


class YOLOv3Loss(nn.Module):
    def __init__(self, anchors, num_classes, box_weight=1., object_weight=1.,
                 no_object_weight=.5, class_weight=1., ignore_iou=.5, ignore_ioa=.5):
        super().__init__()
        self.register_buffer('anchors',anchors.detach().clone().float())
        self.num_classes = num_classes
        self.box_weight,self.object_weight = box_weight,object_weight
        self.no_object_weight,self.class_weight = no_object_weight,class_weight
        self.ignore_iou,self.ignore_ioa = ignore_iou,ignore_ioa

    @torch.no_grad()
    def assign(self,predictions,targets,image_size):
        device = predictions[0].device
        height,width = image_size
        pixel_scale = torch.tensor([width,height],device=device,dtype=torch.float32)
        anchors = self.anchors.to(device)
        maps = []
        for p in predictions:
            shape = p.shape[:-1]
            maps.append(dict(positive=torch.zeros(shape,device=device,dtype=torch.bool),
                             xy=torch.zeros((*shape,2),device=device),
                             wh=torch.zeros((*shape,2),device=device),
                             weight=torch.zeros(shape,device=device),
                             classes=torch.zeros((*shape,self.num_classes),device=device)))
        if not len(targets):
            return maps,0
        wh = targets[:,4:6]*pixel_scale
        flat = anchors.reshape(-1,2)
        inter = torch.minimum(wh[:,None,:],flat[None,:,:]).prod(dim=-1)
        scores = inter/(wh.prod(-1)[:,None]+flat.prod(-1)[None,:]-inter).clamp_min(1e-9)
        # Assign stronger matches first; try another anchor if the same slot is occupied.
        order = scores.max(dim=1).values.argsort(descending=True).tolist()
        skipped = 0
        for index in order:
            b,c,x,y,w,h = targets[index].tolist()
            b,c = int(b),int(c)
            assigned = False
            for anchor_id in scores[index].argsort(descending=True).tolist():
                scale,a = divmod(anchor_id,3)
                gh,gw = predictions[scale].shape[2:4]
                gx,gy = x*gw,y*gh
                j,i = min(int(gx),gw-1),min(int(gy),gh-1)
                current = maps[scale]
                if current['positive'][b,a,i,j]:
                    continue
                current['positive'][b,a,i,j] = True
                current['xy'][b,a,i,j] = torch.tensor([gx-j,gy-i],device=device)
                current['wh'][b,a,i,j] = (wh[index]/anchors[scale,a]).clamp_min(1e-9).log()
                current['weight'][b,a,i,j] = 2-w*h
                current['classes'][b,a,i,j,c] = 1
                assigned = True
                break
            skipped += int(not assigned)
        return maps,skipped

    @torch.no_grad()
    def negative_mask(self,p,anchor_group,targets,ignore_boxes,image_size,positive):
        batch,_,gh,gw,_ = p.shape
        height,width = image_size
        yy,xx = torch.meshgrid(torch.arange(gh,device=p.device),
                               torch.arange(gw,device=p.device),indexing='ij')
        grid = torch.stack((xx,yy),dim=-1).float()
        centers = (p[..., :2].detach().sigmoid()+grid)/p.new_tensor([gw,gh])
        sizes = p[..., 2:4].detach().clamp(-10,10).exp()*anchor_group[:,None,None,:]/p.new_tensor([width,height])
        decoded = xywh_to_xyxy(torch.cat((centers,sizes),dim=-1)).reshape(batch,-1,4)
        negative = ~positive.clone()
        for b in range(batch):
            truth = xywh_to_xyxy(targets[targets[:,0] == b,2:6])
            ignores = ignore_boxes[b].to(device=p.device,dtype=p.dtype)
            keep = negative[b].reshape(-1)
            for start in range(0,len(decoded[b]),4096):
                boxes = decoded[b,start:start+4096]
                end = start+len(boxes)
                if len(truth):
                    keep[start:end] &= pairwise_overlap(boxes,truth).max(dim=1).values < self.ignore_iou
                if len(ignores):
                    keep[start:end] &= pairwise_overlap(boxes,ignores,ioa=True).max(dim=1).values < self.ignore_ioa
        return negative

    def forward(self,predictions,targets,ignore_boxes,image_size):
        if len(predictions) != 3:
            raise ValueError('Expected three YOLOv3 output scales')
        batch = predictions[0].shape[0]
        if len(ignore_boxes) != batch or targets.ndim != 2 or targets.shape[1] != 6:
            raise ValueError('Expected batch ignore lists and targets [N,6]')
        if any(p.shape[0] != batch or p.shape[1] != 3 or p.shape[-1] != 5+self.num_classes for p in predictions):
            raise ValueError('Output shapes do not match class count/batch')
        # Run loss in float32 even if AMP is introduced later.
        predictions = [p.float() for p in predictions]
        targets = targets.to(device=predictions[0].device,dtype=torch.float32)
        if len(targets):
            if not torch.isfinite(targets).all() or not ((targets[:,:2] == targets[:,:2].floor()).all() and
                    (targets[:,0] >= 0).all() and (targets[:,0] < batch).all() and
                    (targets[:,1] >= 0).all() and (targets[:,1] < self.num_classes).all() and
                    (targets[:,2:4] >= 0).all() and (targets[:,2:4] <= 1).all() and
                    (targets[:,4:6] > 0).all() and (targets[:,4:6] <= 1).all()):
                raise ValueError('Invalid target batch/classes/coordinates')
        maps,skipped = self.assign(predictions,targets,image_size)
        zero = sum(p.sum()*0 for p in predictions)
        box_loss = pos_obj = neg_obj = class_loss = zero
        positives = negatives = 0
        for scale,(p,m) in enumerate(zip(predictions,maps)):
            pos = m['positive']
            neg = self.negative_mask(p,self.anchors[scale].to(p.device),targets,ignore_boxes,image_size,pos)
            n_pos,n_neg = int(pos.sum()),int(neg.sum())
            positives += n_pos; negatives += n_neg
            if n_pos:
                xy = F.binary_cross_entropy_with_logits(p[..., :2][pos],m['xy'][pos],reduction='none').sum(-1)
                wh = .5*F.mse_loss(p[..., 2:4][pos],m['wh'][pos],reduction='none').sum(-1)
                box_loss = box_loss+((xy+wh)*m['weight'][pos]).sum()
                pos_obj = pos_obj+F.binary_cross_entropy_with_logits(p[...,4][pos],torch.ones_like(p[...,4][pos]),reduction='sum')
                class_loss = class_loss+F.binary_cross_entropy_with_logits(p[...,5:][pos],m['classes'][pos],reduction='sum')
            if n_neg:
                neg_obj = neg_obj+F.binary_cross_entropy_with_logits(p[...,4][neg],torch.zeros_like(p[...,4][neg]),reduction='sum')
        box_loss = box_loss/max(positives,1)
        pos_obj = pos_obj/max(positives,1)
        neg_obj = neg_obj/max(negatives,1)
        class_loss = class_loss/max(positives*self.num_classes,1)
        total = self.box_weight*box_loss+self.object_weight*pos_obj+self.no_object_weight*neg_obj+self.class_weight*class_loss
        stats = dict(loss=float(total.detach()),box=float(box_loss.detach()),
                     object=float(pos_obj.detach()),background=float(neg_obj.detach()),
                     classification=float(class_loss.detach()),positives=positives,
                     unassigned_targets=skipped)
        return total,stats
