import torch
import torch.nn as nn
EPS = 1e-5


def quantize_activation_per_tensor_asym(inps, bits=8):
    inps_res = inps.clone()
    maxq = torch.tensor(2 ** bits - 1)
    xmax = torch.max(inps)
    xmin = torch.min(inps)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(inps / scale) + zero, 0, maxq)
    inps_res = scale * (q - zero)
    return inps_res

def quantize_activation_per_token_asym(inps, bits=8):
    inps_res = torch.zeros_like(inps)
    maxq = torch.tensor(2 ** bits - 1)
    xmax = torch.max(inps, dim=1, keepdim=True)[0]
    xmin = torch.min(inps, dim=1, keepdim=True)[0]
    scale = (xmax - xmin) / maxq
    scale = torch.clamp(scale, min=EPS, max=1e4)
    zero = torch.round(-xmin / scale)
    inps_res = inps / scale
    q = torch.clamp(torch.round(inps_res) + zero, 0, maxq)
    inps_res = scale * (q - zero)
    return inps_res


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
            self,
            bits, perchannel=False, sym=True,
            mse=False, norm=2.4, grid=100, maxshrink=.8
        ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class ActQuantWrapper(nn.Module):

    def __init__(self, module):
        super(ActQuantWrapper, self).__init__()
        self.module = module
        shape = [1] * len(self.module.weight.shape)
        if len(shape) == 4:
            shape[1] = self.module.weight.shape[1]
        if len(shape) == 3:
            shape[2] = self.module.weight.shape[2]
        if len(shape) == 2:
            shape[1] = self.module.weight.shape[1]
        self.quantizer = Quantizer(shape=shape)

    def forward(self, x):
        return self.module(self.quantizer.quantize(x))

def add_actquant(module, name='', layers=[nn.Conv2d, nn.Linear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)

import time

class Quant4Linear(nn.Module):

    def __init__(self, linear, scales, zeros):
        super().__init__()
        self.register_buffer('zeros', zeros.clone() * scales)
        self.register_buffer('scales', scales)
        self.register_buffer('bias', linear.bias.data)
        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        self.register_buffer('qweight', torch.zeros(
            (intweight.shape[0] // 8, intweight.shape[1]), dtype=torch.int, device=self.bias.device
        ))
        for i in range(intweight.shape[0]):
            self.qweight[i // 8] |= intweight[i] << (4 * (i % 8))
        # self.linear = linear.to(torch.device('cuda:0'))

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.zeros)
            # y = self.linear(x)
            return y.reshape(outshape)
        print(x.shape)
        raise ValueError('Only supports a single token currently.')

def make_quant4(module, quantizers, name=''):
    if isinstance(module, Quant4Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers:
            setattr(
                module, attr,
                Quant4Linear(tmp, quantizers[name1].scale, quantizers[name1].zero)
            )
    for name1, child in module.named_children():
        make_quant4(child, quantizers, name + '.' + name1 if name != '' else name1)

class SmoothAndQuantLinear(nn.Module): 

    def __init__(self, in_features, out_features, bias=True, device=torch.device('cuda'), smooth=False, alpha=0.50, min=0.01, act_quant=False, act_bits=8, act_pertoken=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', torch.randn(self.out_features, self.in_features, dtype=torch.float16, requires_grad=False, device=device))
        if bias:
            self.register_buffer('bias', torch.zeros((1, self.out_features), dtype=torch.float16, requires_grad=False, device=device))
        else:
            self.register_buffer('bias', None)
        self.dev = self.weight.device
        self.smooth = smooth
        self.act_smoothed = False
        self.alpha = alpha
        self.min = min
        self.act_quant = act_quant
        self.act_pertoken = act_pertoken
        self.act_bits = act_bits
        self.smooth_scales = None 
        self.act_scales = None 
        self.weight_scales = None

    # def configure(self, smooth=False, alpha=0.50, min=0.01, act_quant=False, pertoken=False, act_bits=8):
    #     self.smooth = smooth
    #     self.alpha = alpha
    #     self.min = min
    #     self.act_quant = act_quant
    #     self.pertoken = pertoken
    #     self.act_bits = act_bits

    @staticmethod
    def from_float(module, smooth=False, alpha=0.50, min=0.01, act_quant=False, act_bits=8, act_pertoken=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = SmoothAndQuantLinear(module.in_features, module.out_features, module.bias is not None, device=module.weight.device, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        new_module.weight = module.weight
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def forward(self, x):
        # smooth
        if self.smooth:
            # smooth scales
            if self.act_scales is not None and self.smooth_scales is None: 
                # weight scales
                if self.weight_scales is None:
                    self.weight_scales = self.weight.abs().max(dim=0)[0]
                self.smooth_scales = (self.act_scales.pow(self.alpha) / self.weight_scales.pow(1-self.alpha)).clamp(min=self.min)
            if self.smooth_scales is not None and not self.act_smoothed:
                x = x/self.smooth_scales # smooth

        # quant
        if self.act_quant:
            if self.act_pertoken:
                x = quantize_activation_per_token_asym(x, self.act_bits)
            else:
                x = quantize_activation_per_tensor_asym(x, self.act_bits)

        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y

