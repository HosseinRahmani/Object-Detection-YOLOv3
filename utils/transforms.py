"""Deterministic preprocessing; augmentation will be added separately."""
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class TrainingAugment:
    """Random transforms on original RGB images; boxes share every spatial transform.

    Inputs/outputs: normalized class+xywh targets and absolute xyxy ignore boxes.
    Objects made too small or too truncated are retained as ignore regions when
    still visible, to avoid incorrectly treating them as background.
    """
    defaults = dict(horizontal_flip=0.5, color_probability=0.8, brightness=0.2,
                    contrast=0.2, saturation=0.2, affine_probability=0.5,
                    scale=0.1, translate=0.05, min_visibility=0.25, min_box_pixels=2.)

    def __init__(self, settings=None):
        settings = dict(settings or {})
        unknown = set(settings)-set(self.defaults)-{'enabled'}
        if unknown:
            raise ValueError(f'Unknown augmentation settings: {sorted(unknown)}')
        self.settings = dict(self.defaults)
        self.settings.update({k: float(v) for k, v in settings.items() if k != 'enabled'})
        import math
        for key, value in self.settings.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
            if key in ('horizontal_flip', 'color_probability', 'affine_probability', 'min_visibility',
                       'brightness', 'contrast', 'saturation', 'translate') and value > 1:
                raise ValueError(f'{key} must be between 0 and 1')
        if self.settings['scale'] >= 1:
            raise ValueError('scale must be less than 1')

    def __call__(self, image, targets, ignore_boxes, rng):
        image = image.convert('RGB')
        width, height = image.size
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 5).copy()
        ignores = np.asarray(ignore_boxes, dtype=np.float32).reshape(-1, 4).copy()
        boxes = np.empty((len(targets), 4), dtype=np.float32)
        if len(targets):
            centers = targets[:, 1:3]*[width, height]
            sizes = targets[:, 3:5]*[width, height]
            boxes[:, :2], boxes[:, 2:] = centers-sizes/2, centers+sizes/2
        c = self.settings
        info = dict(horizontal_flip=False, scale=1., translation=(0., 0.))
        if rng.random() < c['horizontal_flip']:
            image = ImageOps.mirror(image)
            for b in (boxes, ignores):
                b[:, [0, 2]] = width-b[:, [2, 0]]
            info['horizontal_flip'] = True
        if rng.random() < c['affine_probability']:
            scale = rng.uniform(1-c['scale'], 1+c['scale'])
            tx = rng.uniform(-c['translate'], c['translate'])*width
            ty = rng.uniform(-c['translate'], c['translate'])*height
            ox, oy = (1-scale)*width/2+tx, (1-scale)*height/2+ty
            image = image.transform((width, height), Image.Transform.AFFINE,
                                    (1/scale, 0, -ox/scale, 0, 1/scale, -oy/scale),
                                    resample=Image.Resampling.BILINEAR, fillcolor=(114, 114, 114))
            for b in (boxes, ignores):
                b[:, [0, 2]] = b[:, [0, 2]]*scale+ox
                b[:, [1, 3]] = b[:, [1, 3]]*scale+oy
            info.update(scale=scale, translation=(tx, ty))
        # Clip all boxes to the original canvas.
        unclipped = boxes.copy()
        for b in (boxes, ignores):
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
        sizes = boxes[:, 2:]-boxes[:, :2]
        original_sizes = unclipped[:, 2:]-unclipped[:, :2]
        visibility = np.prod(sizes, axis=1)/np.maximum(np.prod(original_sizes, axis=1), 1e-12)
        # Do not filter already-small KITTI objects when no affine was applied.
        affine_applied = info['scale'] != 1 or info['translation'] != (0., 0.)
        keep = (sizes > 0).all(axis=1)
        if affine_applied:
            keep &= (sizes >= c['min_box_pixels']).all(axis=1) & (visibility >= c['min_visibility'])
        visible_dropped = boxes[~keep & (sizes > 0).all(axis=1)]
        ignores = ignores[(ignores[:, 2:] > ignores[:, :2]).all(axis=1)]
        ignores = np.concatenate((ignores, visible_dropped), axis=0).astype(np.float32)
        targets, boxes = targets[keep], boxes[keep]
        targets[:, 1:3] = ((boxes[:, :2]+boxes[:, 2:])/2)/[width, height]
        targets[:, 3:5] = (boxes[:, 2:]-boxes[:, :2])/[width, height]
        if rng.random() < c['color_probability']:
            for key, enhancer in [('brightness', ImageEnhance.Brightness),
                                  ('contrast', ImageEnhance.Contrast),
                                  ('saturation', ImageEnhance.Color)]:
                factor = rng.uniform(1-c[key], 1+c[key])
                image = enhancer(image).enhance(factor)
                info[key] = factor
        info['dropped_objects'] = int((~keep).sum())
        return image, targets, ignores, info


def letterbox(image, targets, ignore_boxes, image_size=416):
    """Resize/pad RGB image and transform boxes.

    targets: float array (N, 5), [class_id, xc, yc, w, h], normalized.
    ignore_boxes: float array (M, 4), absolute [x1,y1,x2,y2].
    Returns RGB uint8 HWC image, normalized targets, normalized xyxy ignores,
    and geometry metadata for mapping predictions back to the source image.
    """
    if not isinstance(image_size, int) or image_size <= 0 or image_size % 32:
        raise ValueError('image_size must be a positive multiple of 32')
    image = image.convert('RGB')
    width, height = image.size
    ratio = min(image_size/width, image_size/height)
    new_width = max(1, min(image_size, round(width*ratio)))
    new_height = max(1, min(image_size, round(height*ratio)))
    sx, sy = new_width/width, new_height/height
    left, top = (image_size-new_width)//2, (image_size-new_height)//2
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (image_size, image_size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 5).copy()
    if len(targets):
        targets[:, 1] = (targets[:, 1]*new_width+left)/image_size
        targets[:, 2] = (targets[:, 2]*new_height+top)/image_size
        targets[:, 3] *= new_width/image_size
        targets[:, 4] *= new_height/image_size
    ignore_boxes = np.asarray(ignore_boxes, dtype=np.float32).reshape(-1, 4).copy()
    if len(ignore_boxes):
        ignore_boxes[:, [0, 2]] = (ignore_boxes[:, [0, 2]]*sx+left)/image_size
        ignore_boxes[:, [1, 3]] = (ignore_boxes[:, [1, 3]]*sy+top)/image_size
    meta = dict(original_size=(height, width), resized_size=(new_height, new_width),
                input_size=(image_size, image_size), scale=(sx, sy), padding=(left, top))
    return np.asarray(canvas).copy(), targets, ignore_boxes, meta


def normalize_image(image):
    """Convert RGB uint8 HWC to contiguous float32 CHW in [0,1]."""
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)/255.0
