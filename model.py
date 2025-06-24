import numpy as np
import math
import os
from torch.distributions.uniform import Uniform
from video.entropy_models import Entropy_bottleneck,Distribution_for_entropy
from torch.autograd import Function
from torch import Tensor
from video.utils import *
import time
from ptflops import get_model_complexity_info
from thop import profile
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

out_channel_N = 64
out_channel_M = 96
out_channel_mv = 128

import torch.nn as nn
import torch.nn.functional as F
import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

def quantize_ste(x: Tensor) -> Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """
    return (torch.round(x) - x).detach() + x

class QReLU(Function):
    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2**bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
            torch.exp(
                (-ctx.alpha**ctx.beta)
                * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
            )
            * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = np.random.uniform(-1, 1)
        # b = 0
        uniform_distribution = Uniform(-0.5 * torch.ones(x.size())
                                       * (2 ** b), 0.5 * torch.ones(x.size()) * (2 ** b)).sample().cuda()
        return torch.round(x + uniform_distribution) - uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g


class Encoder(nn.Sequential):
    def __init__(
            self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
    ):
        super().__init__(
            conv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            #SqueezeExcite(mid_planes),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, out_planes, kernel_size=5, stride=2),
            #SqueezeExcite(out_planes),
        )
class Encoder2(nn.Sequential):
    def __init__(
            self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
    ):
        super().__init__(
            conv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            #SqueezeExcite(mid_planes),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, out_planes, kernel_size=5, stride=2),
            #SqueezeExcite(out_planes),
        )

class Decoder(nn.Sequential):
    def __init__(
            self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
    ):
        super().__init__(
            #SqueezeExcite(in_planes),
            deconv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            #SqueezeExcite(mid_planes),
            deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, out_planes, kernel_size=5, stride=2),
        )


class HyperEncoder(nn.Sequential):
    def __init__(
            self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
    ):
        super().__init__(
            conv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
        )


class HyperDecoder(nn.Sequential):
    def __init__(
            self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
    ):
        super().__init__(
            deconv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, out_planes, kernel_size=5, stride=2),

        )


class HyperDecoderWithQReLU(nn.Module):
    def __init__(
            self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
    ):
        super().__init__()

        def qrelu(input, bit_depth=8, beta=100):
            return QReLU.apply(input, bit_depth, beta)

        self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
        self.qrelu1 = qrelu
        self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
        self.qrelu2 = qrelu
        self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
        self.qrelu3 = qrelu

    def forward(self, x):
        x = self.qrelu1(self.deconv1(x))
        x = self.qrelu2(self.deconv2(x))
        x = self.qrelu3(self.deconv3(x))

        return x
class Hyperprior(nn.Module):
    def __init__(self, planes: int = 192, mid_planes: int = 192):
         super().__init__()
         self.entropy_bottleneck = Entropy_bottleneck(mid_planes)
         self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
         self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
         self.hyper_decoder_scale = HyperDecoderWithQReLU(
                   planes, mid_planes, planes)
         self.gaussian_conditional = Distribution_for_entropy()

    def forward(self, y):
       z = self.hyper_encoder(y)
       z_hat, z_likelihoods = self.entropy_bottleneck(z,1)
       scales = self.hyper_decoder_scale(z_hat)
       means = self.hyper_decoder_mean(z_hat)
       combined = torch.cat((means, scales), dim=1)
       _, y_likelihoods = self.gaussian_conditional(y, combined)
       y_hat = quantize_ste(y - means) + means
       return y_hat, {"y": y_likelihoods, "z": z_likelihoods}



class ScaleSpaceFlow(nn.Module):
    def __init__(self, num_levels: int = 5, sigma0: float = 1.5, scale_field_shift: float = 1.0):
        super(ScaleSpaceFlow, self).__init__()
        self.num_levels = num_levels
        self.sigma0 = sigma0
        self.scale_field_shift = scale_field_shift
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False
        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        self.ConvolutionNet = ConvolutionNet()
        self.motion_hyperprior = Hyperprior()
        self.res_encoder = Encoder(3)
        self.res_decoder = Decoder(3,in_planes=384)
        self.res_hyperprior = Hyperprior()
    def gaussian_volume(self,x, sigma: float, num_levels: int):  # 高斯金字塔
            k = 2 * int(math.ceil(3 * sigma)) + 1  # 卷积核的大小通常取决于标准差的倍数，常见的做法是取标准差的 3 倍向上取整作为卷积核的大小，并确保它是奇数，以便有一个中心像素。
            device = x.device
            dtype = x.dtype if torch.is_floating_point(x) else torch.float32

            kernel = gaussian_kernel2d(k, sigma, device=device,
                                       dtype=dtype)  # 生成了一个二维高斯核 kernel。该高斯核的大小由 k 确定，标准差由 sigma 确定。device 和 dtype 参数用于确保生成的高斯核与输入张量相匹配。
            volume = [x.unsqueeze(2)]  # b,c,1,h,w    第一层

            x = gaussian_blur(x, kernel=kernel)  # 高斯模糊，卷积核在图像上滑动，对每个像素及其周围像素进行加权平均，从而实现对图像的模糊处理。去除图像中的噪声、减少图像的细节和平滑图像
            volume += [x.unsqueeze(2)]  # 第二层
            for i in range(1, num_levels):  # 循环四层 不包含5，1 2 3 4
                x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))  # 下采样
                x = gaussian_blur(x, kernel=kernel)  # 高斯模糊
                interp = x
                for _ in range(0, i):
                    interp = F.interpolate(
                        interp, scale_factor=2, mode="bilinear", align_corners=False
                    )  # 上采样恢复尺寸
                volume.append(interp.unsqueeze(2))  # 将这个经过扩展后的张量添加到 volume 列表中
            return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:  # 维度不等于5
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)  # 两个二维张量，分别表示网格的横坐标和纵坐标 grid.size(H,W,2)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()  # 对每个网格坐标点进行偏移，得到更新后的网格坐标
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(
            1)  # 在最后一个维度进行拼接得到，（B,H,W,C+2），最后增加一个维度，（B,1,H,W,.
        # ）

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )  # 将volume和volume_grid 作为输入。padding_mode 参数指定了填充模式，而 align_corners 数指定了是否对齐角点。函数的作用是根据volume_grid中的坐标来对volume进行空间采样，得到输出out。
        return out.squeeze(2)  # [N, C, H, W]
    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)
        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def forward(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        num_pixels = x_cur.size()[0] * x_cur.size()[1] * x_cur.size()[2] * x_cur.size()[3]

        y_motion = self.motion_encoder(x)
        y_motion_hat, motion_likelihoods = self.motion_hyperprior(y_motion)
        motion_info = self.motion_decoder(y_motion_hat)

        x_pred = self.forward_prediction(x_ref, motion_info)
        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        #encoding_time = time.time() - start_time
        #start_time = time.time()
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat
        #decoding_time = time.time() - start_time
        res_y_likelihoods =res_likelihoods["y"]
        res_z_likelihoods = res_likelihoods["z"]
        mv_y_likelihoods = motion_likelihoods["y"]
        mv_z_likelihoods = motion_likelihoods["z"]
        motion_likelihoods_sum = (torch.sum(torch.log(mv_y_likelihoods)) + torch.sum(
            torch.log(mv_z_likelihoods))) / (-np.log(2) * num_pixels)
        res_likelihoods_sum = (torch.sum(torch.log(res_y_likelihoods)) + torch.sum(
            torch.log(res_z_likelihoods))) / (-np.log(2) * num_pixels)
        likehood_sum = motion_likelihoods_sum + res_likelihoods_sum
        x_rec_list = self.ConvolutionNet(x_rec)
        output_dict = {
            "output": x_rec_list,
            "bppb": likehood_sum,
        }
        #print(f"Encoding Time: {encoding_time:.6f} seconds")
        #print(f"Decoding Time: {decoding_time:.6f} seconds")
        return output_dict



class ConvolutionNet(nn.Module):
        def __init__(self):
            super(ConvolutionNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=5, stride=2, padding=2)
            self.relu = nn.ReLU()

        def forward(self, x):
            out1 = self.relu(self.conv1(x))
            out2 = self.relu(self.conv2(out1))
            out3 = self.relu(self.conv3(out2))
            return [x, out1, out2, out3]

class ScaleSpaceFlow2(nn.Module):
    def __init__(self, num_levels: int = 5, sigma0: float = 1.5, scale_field_shift: float = 1.0):
        super(ScaleSpaceFlow2, self).__init__()
        self.num_levels = num_levels
        self.sigma0 = sigma0
        self.scale_field_shift = scale_field_shift
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False
        self.motion_encoder = Encoder(2 * 5)
        self.motion_decoder = Decoder(2 + 1)
        self.ConvolutionNet = ConvolutionNet()
        self.motion_hyperprior = Hyperprior()
        self.res_encoder = Encoder(3)
        self.res_decoder = Decoder(3,in_planes=384)
        self.res_hyperprior = Hyperprior()
    def gaussian_volume(self,x, sigma: float, num_levels: int):
            k = 2 * int(math.ceil(3 * sigma)) + 1
            device = x.device
            dtype = x.dtype if torch.is_floating_point(x) else torch.float32

            kernel = gaussian_kernel2d(k, sigma, device=device,
                                       dtype=dtype)
            volume = [x.unsqueeze(2)]

            x = gaussian_blur(x, kernel=kernel)
            volume += [x.unsqueeze(2)]
            for i in range(1, num_levels):
                x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
                x = gaussian_blur(x, kernel=kernel)
                interp = x
                for _ in range(0, i):
                    interp = F.interpolate(
                        interp, scale_factor=2, mode="bilinear", align_corners=False
                    )
                volume.append(interp.unsqueeze(2))
            return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(
            1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)  # [N, C, H, W]
    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)
        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def forward(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        num_pixels = x_cur.size()[0] * x_cur.size()[1] * x_cur.size()[2] * x_cur.size()[3]

        y_motion = self.motion_encoder(x)

        y_motion_hat, motion_likelihoods = self.motion_hyperprior(y_motion)

        motion_info = self.motion_decoder(y_motion_hat)
        #start_time = time.time()
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        #encoding_time = time.time() - start_time
        #start_time = time.time()
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat
        #decoding_time = time.time() - start_time
        res_y_likelihoods =res_likelihoods["y"]
        res_z_likelihoods = res_likelihoods["z"]
        mv_y_likelihoods = motion_likelihoods["y"]
        mv_z_likelihoods = motion_likelihoods["z"]
        motion_likelihoods_sum = (torch.sum(torch.log(mv_y_likelihoods)) + torch.sum(
            torch.log(mv_z_likelihoods))) / (-np.log(2) * num_pixels)
        res_likelihoods_sum = (torch.sum(torch.log(res_y_likelihoods)) + torch.sum(
            torch.log(res_z_likelihoods))) / (-np.log(2) * num_pixels)
        likehood_sum = motion_likelihoods_sum + res_likelihoods_sum
        x_rec_list = self.ConvolutionNet(x_rec)
        output_dict = {
            "output": x_rec_list,
            "bppb": likehood_sum,
        }
        #print(f"Encoding Time: {encoding_time:.6f} seconds")
        #print(f"Decoding Time: {decoding_time:.6f} seconds")
        return output_dict

