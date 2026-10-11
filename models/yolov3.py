"""Original YOLOv3 architecture in PyTorch (Darknet-53, three detection scales).

Architecture reference: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
Paper: Redmon & Farhadi, YOLOv3: An Incremental Improvement (2018).
Returns raw predictions; loss, decoding and NMS belong to later pipeline steps.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = ConvBNLeaky(channels, channels//2, 1)
        self.expand = ConvBNLeaky(channels//2, channels, 3)

    def forward(self, x):
        return x+self.expand(self.reduce(x))


class DarknetStage(nn.Module):
    def __init__(self, in_channels, channels, repeats):
        super().__init__()
        self.downsample = ConvBNLeaky(in_channels, channels, 3, 2)
        self.residuals = nn.Sequential(*(Residual(channels) for _ in range(repeats)))

    def forward(self, x):
        return self.residuals(self.downsample(x))


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNLeaky(3, 32, 3)
        self.stages = nn.ModuleList([DarknetStage(32,64,1), DarknetStage(64,128,2),
                                    DarknetStage(128,256,8), DarknetStage(256,512,8),
                                    DarknetStage(512,1024,4)])

    def forward(self, x):
        x = self.stem(x)
        routes = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index >= 2:
                routes.append(x)
        return routes  # strides 8,16,32, with 256,512,1024 channels


class DetectionBlock(nn.Module):
    def __init__(self, in_channels, channels, classes):
        super().__init__()
        self.route = nn.Sequential(ConvBNLeaky(in_channels,channels,1),
                                   ConvBNLeaky(channels,channels*2,3),
                                   ConvBNLeaky(channels*2,channels,1),
                                   ConvBNLeaky(channels,channels*2,3),
                                   ConvBNLeaky(channels*2,channels,1))
        self.features = ConvBNLeaky(channels,channels*2,3)
        self.prediction = nn.Conv2d(channels*2,3*(5+classes),1,bias=True)

    def forward(self, x):
        route = self.route(x)
        return route, self.prediction(self.features(route))


class YOLOv3(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        if not isinstance(num_classes,int) or num_classes < 1:
            raise ValueError('num_classes must be a positive integer')
        self.num_classes = num_classes
        self.backbone = Darknet53()
        # Registration order follows the convolution order in the Darknet file.
        self.large = DetectionBlock(1024,512,num_classes)
        self.large_route = ConvBNLeaky(512,256,1)
        self.medium = DetectionBlock(768,256,num_classes)
        self.medium_route = ConvBNLeaky(256,128,1)
        self.small = DetectionBlock(384,128,num_classes)
        self.register_buffer('anchors', torch.tensor([
            [[116,90],[156,198],[373,326]],
            [[30,61],[62,45],[59,119]],
            [[10,13],[16,30],[33,23]],
        ],dtype=torch.float32))
        # Anchor widths/heights in input pixels; original defaults kept initially.
        self.register_buffer('strides',torch.tensor([32.,16.,8.]))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected [batch,3,height,width]')
        if images.shape[2] % 32 or images.shape[3] % 32:
            raise ValueError('Input height and width must be multiples of 32')
        stride8,stride16,stride32 = self.backbone(images)
        large_route, large = self.large(stride32)
        middle = F.interpolate(self.large_route(large_route),scale_factor=2,mode='nearest')
        medium_route, medium = self.medium(torch.cat((middle,stride16),dim=1))
        fine = F.interpolate(self.medium_route(medium_route),scale_factor=2,mode='nearest')
        _,small = self.small(torch.cat((fine,stride8),dim=1))
        outputs = []
        for prediction in (large,medium,small):
            batch,_,height,width = prediction.shape
            outputs.append(prediction.reshape(batch,3,5+self.num_classes,height,width)
                           .permute(0,1,3,4,2).contiguous())
        return outputs  # [B,3,H,W,5+C], in stride 32,16,8 order; no sigmoid applied

    def darknet_layers(self):
        """Yield each convolution once, in original Darknet serialization order."""
        for module in self.modules():
            if isinstance(module,ConvBNLeaky):
                yield module.conv,module.bn
            elif isinstance(module,nn.Conv2d) and module.bias is not None:
                yield module,None

    @torch.no_grad()
    def load_darknet_weights(self, path, source_classes=80):
        """Load standard full yolov3.weights; skip incompatible prediction layers.

        Every source float is consumed, including the three skipped convolutions,
        so subsequent layers stay correctly aligned. Tiny/SPP and backbone-only
        files are incompatible. Does not load Ultralytics .pt checkpoints.
        """
        layers = list(self.darknet_layers())
        if len(layers) != 75:
            raise RuntimeError('Expected 75 convolutions in original YOLOv3')
        path = Path(path)
        with path.open('rb') as file:
            version = np.fromfile(file,dtype='<i4',count=3)
            if len(version) != 3:
                raise ValueError('Truncated Darknet header')
            major,minor,_ = map(int,version)
            if not (0 <= major < 1000 and 0 <= minor < 1000):
                raise ValueError('Unsupported Darknet version')
            seen_dtype = '<i8' if major*10+minor >= 2 else '<i4'
            seen = np.fromfile(file,dtype=seen_dtype,count=1)
            if len(seen) != 1:
                raise ValueError('Missing Darknet seen counter')
            expected = sum((4*conv.out_channels if bn is not None else 3*(5+source_classes))+
                           (conv.weight.numel() if bn is not None else
                            3*(5+source_classes)*conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1])
                           for conv,bn in layers)
            if path.stat().st_size-file.tell() != expected*4:
                raise ValueError(f'Incompatible/truncated file: expected {expected*4} weight bytes')

            def read(count):
                array = np.fromfile(file,dtype='<f4',count=count)
                if len(array) != count or not np.isfinite(array).all():
                    raise ValueError('Truncated or non-finite weights')
                return array

            def copy(parameter, array):
                parameter.copy_(torch.from_numpy(array.copy()).reshape(parameter.shape)
                                .to(device=parameter.device,dtype=parameter.dtype))

            loaded,skipped = 0,0
            for conv,bn in layers:
                if bn is not None:
                    for tensor in (bn.bias,bn.weight,bn.running_mean,bn.running_var):
                        copy(tensor,read(tensor.numel()))
                    copy(conv.weight,read(conv.weight.numel()))
                    loaded += 1
                else:
                    source_outputs = 3*(5+source_classes)
                    bias = read(source_outputs)
                    weight = read(source_outputs*conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1])
                    if self.num_classes == source_classes:
                        copy(conv.bias,bias); copy(conv.weight,weight)
                        loaded += 1
                    else:
                        skipped += 1
            if file.read(1):
                raise ValueError('Unexpected bytes after final layer')
        return dict(loaded_convolutions=loaded, initialized_prediction_layers=skipped,
                    source_classes=source_classes, target_classes=self.num_classes,
                    darknet_version=version.tolist(), seen=int(seen[0]))
