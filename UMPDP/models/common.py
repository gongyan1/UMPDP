 # YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

from typing import Optional
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential
# from timm.models.layers import trunc_normal, DropPath
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)  if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() 

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])
    
class AddC(nn.Module):
    #  Add two tensors
    def __init__(self, arg, c1):
        super(AddC, self).__init__()
        self.arg = arg
        self.conv1 = Conv(2*c1, c1, 1, 1)

    def forward(self, x):
        return self.conv1(torch.cat((x[0], x[1]),dim=1))
    

class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
#             print(type(x[0]), type(x[1][0]))
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
#             print(type(x[0]), type(x[1][1]))
#             print(x[0].shape, x[1][0].shape)
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])
    
def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


class Add3(nn.Module):
    #  Add two tensors
    def __init__(self, c1, arg):
        super(Add3, self).__init__()
        self.arg = arg
        self.conv1 = Conv(2*c1, c1, 1, 1)


    def forward(self, x):
        out = torch.cat((x[0], x[1]), dim=1)
        out = shuffle_channels(out, 2)
        return self.conv1(out)
    

class Add4(nn.Module):
    #  Add two tensors
    def __init__(self, c1, arg):
        super(Add4, self).__init__()
        self.arg = arg
        self.conv1 = Conv(c1, c1//2, 1, 1)
        self.conv2 = Conv(c1, c1//2, 1, 1)


    def forward(self, x):
        out = torch.cat((self.conv1(x[0]), self.conv2(x[1])), dim=1)
        out = shuffle_channels(out, 2)
        return out
    
    
class Add5(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index
        self.conv1 = Conv(2*c1, c1, 1, 1)
        self.conv2 = Conv(2*c1, c1, 1, 1)

    def forward(self, x):
        if self.index == 0:
            sum = torch.cat((x[0], x[1][0]), dim=1)
            return self.conv1(sum)
        elif self.index == 1:
            sum = torch.cat((x[0], x[1][1]), dim=1)
            return self.conv2(sum)

    
        
class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)

class ASFFV5(nn.Module):
    def __init__(self, level, rfb=False, vis=False):

    #  512, 256, 128 -> multiplier=1
    # 256, 128, 64 -> multiplier=0.5

        super(ASFFV5, self).__init__()
        self.level = level
        self.multiplier = 1

        self.dim = [int(1024 * self.multiplier), int(512 * self.multiplier),
                    int(256 * self.multiplier)]


        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * self.multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(256 * self.multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                1024 * self.multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * self.multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(256 * self.multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * self.multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * self.multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * self.multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                256 * self.multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[2]  # l
        x_level_1 = x[1]  # m
        x_level_2 = x[0]  # s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)



class DMAF(nn.Module):
    def __init__(self, c1):
        super(DMAF, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv1 = Conv(c1, 2*c1, 1, 2)
        self.conv1_t = Conv(c1, 2*c1, 1, 2)
        self.conv2 = Conv(c1, 2*c1, 3, 2)
        self.conv2_t = Conv(c1, 2*c1, 3, 2)

    def forward(self, x, y):
        fdx = x - y#N,C,H,W
        vx = F.tanh(self.GAP(fdx))#N,C,1,1
        x_res = x + vx * y
        fdy = y - x
        vy = F.tanh(self.GAP(fdy))
        y_res = y + vy * x
        x = self.conv1(x)
        x_res = self.conv2(x_res)
        y = self.conv1_t(y)
        y_res = self.conv2_t(y_res)

        return x+x_res, y+y_res


class DMAF1(nn.Module):
    def __init__(self, c1):#差分融合调整
        super(DMAF1, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv2 = Conv(c1, c1, 3, 1)
        self.conv2_t = Conv(c1, c1, 3, 1)

    def forward(self, z):
        x, y = z[0], z[1]
        fdx = x - y#N,C,H,W
        vx = F.tanh(self.GAP(fdx))#N,C,1,1
        fdy = y - x
        vy = F.tanh(self.GAP(fdy))
        y_res = y + vx * x
        x_res = x + vy * y
        x_res = self.conv2(x_res)
        y_res = self.conv2_t(y_res)

        return x_res, y_res


class TN(nn.Module):
     def __init__(self, c1):  # c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
         super(TN, self).__init__()
         self.x_tr = Conv(c1, c1, 1, 1)
         self.y_tr = Conv(c1, c1, 1, 1)

         self.alpha = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.alpha1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midy_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True),
             nn.ReLU(inplace=True)
         )

         self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midx_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True),
             nn.ReLU(inplace=True)
         )

     def forward(self, z):
         x, y = z[0], z[1]

         x_tr = self.x_tr(x)
         y_tr = self.y_tr(y)

         xy_cat = torch.cat((x_tr, y_tr), dim=1)

         alpha = self.alpha(xy_cat)
         beta = self.beta(xy_cat)

         alpha1 = self.alpha1(xy_cat)
         beta1 = self.beta1(xy_cat)

         mid_y = self.y_t(y)
         y2x = (alpha + 1) * mid_y + beta
         y2x_feat = self.midx_fuse(y2x)
         x_mix = y2x_feat

         mid_x = self.x_t(x)
         x2y = (alpha1 + 1) * mid_x + beta1
         x2y_feat = self.midy_fuse(x2y)
         y_mix = x2y_feat

         return x_mix, y_mix


class TN_DMAF(nn.Module):
    def __init__(self, c1):
        super(TN_DMAF, self).__init__()
        self.x_tr = Conv(c1, c1, 1, 1)
        self.y_tr = Conv(c1, c1, 1, 1)

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=True)
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv2 = Conv(c1, c1, 3, 1)
        self.conv2_t = Conv(c1, c1, 3, 1)

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
        y2x_feat = self.midy_fuse(y2x)
        x = x + 0.1 * y2x_feat

        fdx = x - y#N,C,H,W
        vx = torch.tanh(self.GAP(fdx))#N,C,1,1
        fdy = y - x
        vy = torch.tanh(self.GAP(fdy))
        x_res = x + vy * y
        y_res = y + vx * x
        x_res = self.conv2(x_res)
        y_res = self.conv2_t(y_res)

        return x_res, y_res


        

class DMAF_SE(nn.Module):
    def __init__(self, c1, reduction=4):
        super(DMAF_SE, self).__init__()
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
    def forward(self, z):
        x, y = z[0], z[1]
        sub_xy = x - y#N,C,H,W
   
        vx = torch.tanh(self.sub1(sub_xy))
        sub_yx = y - x
        vy = torch.tanh(self.sub2(sub_yx))
        
        y_res = y + vx * x
        x_res = x + vy * y

        x_mix = self.conv1(x_res)
        y_mix = self.conv2(y_res)

        return x_mix, y_mix


class DMAF_SE1(nn.Module):
     def __init__(self, c1, reduction=4):
         super(DMAF_SE1, self).__init__()
         self.sub1 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
         )
         self.sub2 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

         )
         self.conv1 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.SiLU()
         )
         self.conv2 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.SiLU()
         )
         self.sig = nn.Sigmoid()

     def forward(self, z):
         x, y = z[0], z[1]
         sub_xy = x - y  # N,C,H,W

         vx = self.sig(self.sub1(sub_xy))
         sub_yx = y - x
         vy = self.sig(self.sub2(sub_yx))

         y_res = y + vx * x
         x_res = x + vy * y

         x_mix = self.conv1(x_res)
         y_mix = self.conv2(y_res)

         return x_mix, y_mix

    
class TN1(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN1, self).__init__()
        self.x_tr = Conv(c1, c1, 1, 1)
        self.y_tr = Conv(c1, c1, 1, 1)

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
        y2x_feat = self.midx_fuse(y2x + x)
        x_mix = y2x_feat
        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1
        x2y_feat = self.midy_fuse(x2y + y)
        y_mix = x2y_feat

        return x_mix, y_mix
    

    
class TN_DMAF7(nn.Module):
    def __init__(self, c1,reduction=2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAF7, self).__init__()
        self.x_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.y_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.convx3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1,1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.convy3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.sig = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
#         y2x = self.relu(y2x)
        y2x_feat = self.midx_fuse(y2x)
        x_mix = self.convx3(y2x_feat + x)
        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1
#         x2y = self.relu(x2y)
        x2y_feat = self.midy_fuse(x2y)
        y_mix = self.convy3(x2y_feat + y)
        
        sub_xy = x - y#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = vy * y + x + x_mix
        x_fusion = self.conv1(x_res)
        
        y_res = vx * x + y + y_mix
        y_fusion = self.conv2(y_res)
        
        return x_fusion, y_fusion


class TN_DMAF_cat2(nn.Module):
     def __init__(self, c1, reduction=2):  # c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
         super(TN_DMAF_cat2, self).__init__()
         self.x_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.y_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.alpha = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.alpha1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midy_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True),

         )

         self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midx_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True),

         )

         self.sub1 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
         )
         self.sub2 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

         )
         self.conv1 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.conv2 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.convx3 = nn.Sequential(
             nn.Conv2d(c1, c1, 3, 1, 1),
             nn.BatchNorm2d(c1),

         )
         self.convy3 = nn.Sequential(
             nn.Conv2d(c1, c1, 3, 1, 1),
             nn.BatchNorm2d(c1),

         )

         self.sig = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)

     def forward(self, z):
         x, y = z[0], z[1]

         x_tr = self.x_tr(x)
         y_tr = self.y_tr(y)

         xy_cat = torch.cat((x_tr, y_tr), dim=1)

         alpha = self.alpha(xy_cat)
         beta = self.beta(xy_cat)

         alpha1 = self.alpha1(xy_cat)
         beta1 = self.beta1(xy_cat)

         mid_y = self.y_t(y)
         y2x = (alpha + 1) * mid_y + beta
         #         y2x = self.relu(y2x)
         y2x_feat = self.midx_fuse(y2x)
         #         x_mix = y2x_feat
         x_mix = self.convx3(y2x_feat + x)

         mid_x = self.x_t(x)
         x2y = (alpha1 + 1) * mid_x + beta1
         #         x2y = self.relu(x2y)
         x2y_feat = self.midy_fuse(x2y)
         #         y_mix = x2y_feat
         y_mix = self.convy3(x2y_feat + y)

         sub_xy = x - y  # N,C,H,W
         vx = self.sig(self.sub1(sub_xy))

         sub_yx = y - x
         vy = self.sig(self.sub2(sub_yx))

         x_res = vy * y + x + x_mix
         x_fusion = self.conv1(x_res)

         y_res = vx * x + y + y_mix
         y_fusion = self.conv2(y_res)

         return x_fusion, y_fusion


class TN_DMAF_cat1(nn.Module):
     def __init__(self, c1, reduction=2):  # c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
         super(TN_DMAF_cat1, self).__init__()
         self.x_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.y_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.alpha = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.alpha1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midy_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True),
         )

         self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midx_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True),
         )

         self.sub1 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
         )
         self.sub2 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

         )
         self.conv1 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.conv2 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.sig = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)

     def forward(self, z):
         x, y = z[0], z[1]

         x_tr = self.x_tr(x)
         y_tr = self.y_tr(y)

         xy_cat = torch.cat((x_tr, y_tr), dim=1)

         alpha = self.alpha(xy_cat)
         beta = self.beta(xy_cat)

         alpha1 = self.alpha1(xy_cat)
         beta1 = self.beta1(xy_cat)

         mid_y = self.y_t(y)
         y2x = (alpha + 1) * mid_y + beta

         y2x_feat = self.midx_fuse(y2x + x)

         mid_x = self.x_t(x)
         x2y = (alpha1 + 1) * mid_x + beta1

         x2y_feat = self.midy_fuse(x2y + y)

         sub_xy = x - y  # N,C,H,W
         vx = self.sig(self.sub1(sub_xy))

         sub_yx = y - x
         vy = self.sig(self.sub2(sub_yx))

         x_res = vy * y + x
         x_fusion = self.conv1(x_res)

         y_res = vx * x + y
         y_fusion = self.conv2(y_res)

         return x_fusion + y2x_feat, y_fusion + x2y_feat
    
class TN_DMAF_cat(nn.Module):
    def __init__(self, c1,reduction=2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAF_cat, self).__init__()
        self.x_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.y_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
        )
        
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        
        self.sig = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta

        y2x_feat = self.midx_fuse(y2x + x)

        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1

        x2y_feat = self.midy_fuse(x2y + y)

        
        sub_xy = x - y#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = vy * y + x + y2x_feat
        x_fusion = self.conv1(x_res)
        
        y_res = vx * x + y + x2y_feat
        y_fusion = self.conv2(y_res)
        
        return x_fusion, y_fusion


class TN_DMAF_cas(nn.Module):
     def __init__(self, c1, reduction=2):  # c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
         super(TN_DMAF_cas, self).__init__()
         self.x_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.y_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.alpha = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.alpha1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.beta1 = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=True)

         self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midy_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True),
             #             nn.ReLU(inplace=True)
         )

         self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midx_fuse = nn.Sequential(
             nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True),
             #             nn.ReLU(inplace=True)
         )

         self.sub1 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
         )
         self.sub2 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

         )
         self.conv1 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.conv2 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.convx3 = nn.Sequential(
             nn.Conv2d(c1, c1, 3, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.convy3 = nn.Sequential(
             nn.Conv2d(c1, c1, 3, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.sig = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)

     def forward(self, z):
         x, y = z[0], z[1]

         x_tr = self.x_tr(x)
         y_tr = self.y_tr(y)

         xy_cat = torch.cat((x_tr, y_tr), dim=1)

         alpha = self.alpha(xy_cat)
         beta = self.beta(xy_cat)

         alpha1 = self.alpha1(xy_cat)
         beta1 = self.beta1(xy_cat)

         mid_y = self.y_t(y)
         y2x = (alpha + 1) * mid_y + beta
         #         y2x = self.relu(y2x)
         y2x_feat = self.midx_fuse(y2x)
         #         x_mix = y2x_feat
         x_mix = self.convx3(y2x_feat + x)

         mid_x = self.x_t(x)
         x2y = (alpha1 + 1) * mid_x + beta1
         #         x2y = self.relu(x2y)
         x2y_feat = self.midy_fuse(x2y)
         #         y_mix = x2y_feat
         y_mix = self.convy3(x2y_feat + y)

         sub_xy = x_mix - y_mix  # N,C,H,W
         vx = self.sig(self.sub1(sub_xy))

         sub_yx = y_mix - x_mix
         vy = self.sig(self.sub2(sub_yx))

         x_res = vy * y_mix + x_mix + y2x_feat
         x_mix = self.conv1(x_res)

         y_res = vx * x_mix + y_mix + x2y_feat
         y_mix = self.conv2(y_res)

         return x_mix, y_mix

    
class TN_DMAFcas1(nn.Module):
    def __init__(self, c1,reduction=2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAFcas1, self).__init__()
        self.x_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.y_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.convx3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1,1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.convy3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.sig = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
#         y2x = self.relu(y2x)
        y2x_feat = self.midx_fuse(y2x)
        x_mix = y2x_feat
        x_mix = self.convx3(x_mix + x)
        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1
#         x2y = self.relu(x2y)
        x2y_feat = self.midy_fuse(x2y)
        y_mix = x2y_feat
        y_mix = self.convy3(y_mix + y)
        
        sub_xy = x_mix - y_mix#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        
        sub_yx = y_mix - x_mix
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = vy * y_mix + x_mix + y2x_feat
        x_mix = self.conv1(x_res)
        
        y_res = vx * x_mix + y_mix + x2y_feat
        y_mix = self.conv2(y_res)
        
        return x_mix , y_mix

class TN_DMAF8_att1(nn.Module):
     def __init__(self, c1,reduction=2):
         super(TN_DMAF8_att1, self).__init__()
         self.x_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.y_tr = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
         self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

         self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
         self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

         self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midy_fuse = nn.Sequential(
             nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
             #             nn.ReLU(inplace=True)
         )

         self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
         self.midx_fuse = nn.Sequential(
             nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
             #             nn.ReLU(inplace=True)
         )

         self.sub1 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
         )
         self.sub2 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),

             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
             nn.ReLU(c1 // reduction),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

         )
         self.conv1 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )
         self.conv2 = nn.Sequential(
             nn.Conv2d(c1, c1, 1, 1),
             nn.BatchNorm2d(c1),
             nn.ReLU()
         )

         self.local_att1 = nn.Sequential(
             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0),
             nn.BatchNorm2d(c1 // reduction),
             nn.ReLU(inplace=True),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0),
             nn.BatchNorm2d(c1),
         )

         self.local_att2 = nn.Sequential(
             nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0),
             nn.BatchNorm2d(c1 // reduction),
             nn.ReLU(inplace=True),
             nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0),
             nn.BatchNorm2d(c1),
         )


         self.sig = nn.Sigmoid()
         self.relu=nn.ReLU(inplace=True)


     def forward(self, z):
         x, y = z[0], z[1]

         x_tr = self.x_tr(x)
         y_tr = self.y_tr(y)

         xy_cat = torch.cat((x_tr, y_tr), dim=1)

         alpha = self.alpha(xy_cat)
         beta = self.beta(xy_cat)

         alpha1 = self.alpha1(xy_cat)
         beta1 = self.beta1(xy_cat)

         mid_y = self.y_t(y)
         y2x = (alpha + 1) * mid_y + beta
         y2x_feat = self.midx_fuse(y2x + x)

         mid_x = self.x_t(x)
         x2y = (alpha1 + 1) * mid_x + beta1
         #
         x2y_feat = self.midy_fuse(x2y + y)


         sub_xy = x - y#N,C,H,W
         vx = self.sig(self.sub1(sub_xy))

         sub_yx = y - x
         vy = self.sig(self.sub2(sub_yx))

         x_res = vy * y + x * self.sig(self.local_att1(y2x_feat)) + x
         x_fusion = self.conv1(x_res)

         y_res = vx * x + y * self.sig(self.local_att2(x2y_feat)) + y
         y_fusion = self.conv2(y_res)

         return x_fusion, y_fusion
