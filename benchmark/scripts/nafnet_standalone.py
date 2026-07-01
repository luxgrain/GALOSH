"""Standalone NAFNet arch (extracted from NAFNet/basicsr to avoid the basicsr+torchvision
import breakage). Plain NAFNet (no Local_Base/TLC) — fine for full-image inference."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(3).sum(2).sum(0), grad_output.sum(3).sum(2).sum(0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, groups=1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, groups=1, bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x); x = self.conv2(x); x = self.sg(x); x = x * self.sca(x); x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y)); x = self.sg(x); x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, groups=1, bias=True)
        self.encoders = nn.ModuleList(); self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList(); self.ups = nn.ModuleList(); self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x); encs.append(x); x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x); x = x + enc_skip; x = decoder(x)
        x = self.ending(x); x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph))


def load_nafnet_sidd(weight_path, device="cuda"):
    m = NAFNet(img_channel=3, width=64, middle_blk_num=12,
               enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ck = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    sd = ck.get("params", ck)
    m.load_state_dict(sd, strict=True)
    return m.eval().to(device)


# ---- TLC (Test-time Local Converter) for NAFNet: replace global AdaptiveAvgPool2d in the
# Simplified Channel Attention with a LOCAL window pool, so test on images larger than the
# 256x256 train patch stays stable (the plain model diverges to +-1000 on large/dark OOD).
class _LocalAvgPool2d(nn.Module):
    def __init__(self, base_size, train_size):
        super().__init__(); self.base_size=base_size; self.train_size=train_size; self.kernel_size=None
    def forward(self, x):
        if self.kernel_size is None:
            bs=self.base_size if isinstance(self.base_size,tuple) else (self.base_size,self.base_size)
            self.kernel_size=[x.shape[2]*bs[0]//self.train_size[-2], x.shape[3]*bs[1]//self.train_size[-1]]
        k0,k1=self.kernel_size
        if k0>=x.size(-2) and k1>=x.size(-1):
            return F.adaptive_avg_pool2d(x,1)
        n,c,h,w=x.shape
        s=x.cumsum(-1).cumsum(-2); s=F.pad(s,(1,0,1,0))
        k0,k1=min(h,k0),min(w,k1)
        out=(s[:,:,:-k0,:-k1]+s[:,:,k0:,k1:]-s[:,:,:-k0,k1:]-s[:,:,k0:,:-k1])/(k0*k1)
        _h,_w=out.shape[2:]
        return F.pad(out,((w-_w)//2,(w-_w+1)//2,(h-_h)//2,(h-_h+1)//2),mode='replicate')

def _apply_tlc(model, base_size, train_size):
    for n,m in model.named_children():
        if len(list(m.children()))>0: _apply_tlc(m,base_size,train_size)
        if isinstance(m,nn.AdaptiveAvgPool2d): setattr(model,n,_LocalAvgPool2d(base_size,train_size))

def load_nafnet_sidd_tlc(weight_path, device="cuda", base_size=384):
    m = load_nafnet_sidd(weight_path, "cpu")
    _apply_tlc(m, base_size, (1,3,256,256))
    return m.eval().to(device)
