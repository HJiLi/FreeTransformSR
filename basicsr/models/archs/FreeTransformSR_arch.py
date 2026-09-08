import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from fvcore.nn import FlopCountAnalysis
    import logging
    logging.getLogger('fvcore').setLevel(logging.ERROR)
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("Tip: Install fvcore for accurate FLOPs calculation: pip install fvcore")


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return to_4d(x, h, w)


class effCA(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class ChannelWiseMixedBasisTransform(nn.Module):
    def __init__(self, dim, window_size=8, rank=1, basis_names=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.rank = rank

        eye = torch.eye(window_size).unsqueeze(0).repeat(dim, 1, 1)
        self.register_buffer("eye", eye, persistent=False)

        self.delta_h_a = nn.Parameter(torch.zeros(dim, window_size, rank))
        self.delta_h_b = nn.Parameter(torch.zeros(dim, rank, window_size))
        self.delta_w_a = nn.Parameter(torch.zeros(dim, window_size, rank))
        self.delta_w_b = nn.Parameter(torch.zeros(dim, rank, window_size))

    def build_bases(self):
        delta_h = torch.matmul(self.delta_h_a, self.delta_h_b)
        delta_w = torch.matmul(self.delta_w_a, self.delta_w_b)
        th = self.eye + delta_h
        tw = self.eye + delta_w
        return th, tw

    def forward_with_bases(self, x, th, tw):
        return torch.einsum('cui,ncij,cvj->ncuv', th, x, tw)

    def inverse_with_bases(self, y, th, tw):
        return torch.einsum('cui,ncuv,cvj->ncij', th, y, tw)

    def forward(self, x):
        th, tw = self.build_bases()
        return self.forward_with_bases(x, th, tw)

    def inverse(self, y):
        th, tw = self.build_bases()
        return self.inverse_with_bases(y, th, tw)


class LearnedTransformBlock(nn.Module):
    def __init__(self, dim, base_window=8, rank=1):
        super().__init__()
        self.base_window = base_window
        self.norm = LayerNorm2d(dim)
        self.projin = nn.Conv2d(dim, dim * 2, 1)
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

        self.transform = ChannelWiseMixedBasisTransform(
            dim=dim,
            window_size=base_window,
            rank=rank
        )

        self.coeff = nn.Parameter(torch.ones(dim, base_window, base_window))
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x0):
        b, c, h, w = x0.shape
        x = self.norm(x0)

        pad_h = (self.base_window - h % self.base_window) % self.base_window
        pad_w = (self.base_window - w % self.base_window) % self.base_window
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        _, _, hp, wp = x.shape

        x1, x2 = self.projin(x).chunk(2, dim=1)
        scale = 0.8 + 0.6 * self.scale_gate(x)
        num_h, num_w = hp // self.base_window, wp // self.base_window

        x1 = rearrange(
            x1,
            'b c (nh dh) (nw dw) -> (b nh nw) c dh dw',
            dh=self.base_window,
            dw=self.base_window
        )

        th, tw = self.transform.build_bases()
        x1 = self.transform.forward_with_bases(x1, th, tw)

        scale = scale.expand(b, 1, num_h, num_w).permute(0, 2, 3, 1).reshape(
            b * num_h * num_w, 1, 1, 1
        )
        x1 = x1 * self.coeff.unsqueeze(0) * scale
        x1 = self.transform.inverse_with_bases(x1, th, tw)

        x1 = rearrange(
            x1,
            '(b nh nw) c dh dw -> b c (nh dh) (nw dw)',
            b=b,
            nh=num_h,
            nw=num_w
        )

        x = self.proj(x1 * self.local(x2))

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        return x + x0


class WindowAttention(nn.Module):
    def __init__(self, dim, heads=3, wsize=16, shift=False):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
        self.dim = dim
        self.heads = heads
        self.wsize = wsize
        self.shift = shift
        self.scale = (dim // heads) ** -0.5
        self.norm = LayerNorm2d(dim)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x0):
        b, c, h, w = x0.shape
        x = self.norm(x0)

        pad_h = (self.wsize - h % self.wsize) % self.wsize
        pad_w = (self.wsize - w % self.wsize) % self.wsize
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        _, _, hp, wp = x.shape

        q = self.q_proj(x)
        k, v = self.kv(x).chunk(2, dim=1)

        if self.shift:
            s = self.wsize // 2
            q = torch.roll(q, shifts=(-s, -s), dims=(2, 3))
            k = torch.roll(k, shifts=(-s, -s), dims=(2, 3))
            v = torch.roll(v, shifts=(-s, -s), dims=(2, 3))

        q = rearrange(q, 'b (hed c) (nh dh) (nw dw) -> (b nh nw) hed (dh dw) c', dh=self.wsize, dw=self.wsize, hed=self.heads)
        k = rearrange(k, 'b (hed c) (nh dh) (nw dw) -> (b nh nw) hed (dh dw) c', dh=self.wsize, dw=self.wsize, hed=self.heads)
        v = rearrange(v, 'b (hed c) (nh dh) (nw dw) -> (b nh nw) hed (dh dw) c', dh=self.wsize, dw=self.wsize, hed=self.heads)

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        y = attn @ v

        y = rearrange(
            y,
            '(b nh nw) hed (dh dw) c -> b (hed c) (nh dh) (nw dw)',
            nh=hp // self.wsize,
            nw=wp // self.wsize,
            dh=self.wsize,
            dw=self.wsize
        )

        if self.shift:
            y = torch.roll(y, shifts=(self.wsize // 2, self.wsize // 2), dims=(2, 3))

        y = self.proj(y)
        if pad_h > 0 or pad_w > 0:
            y = y[:, :, :h, :w]
        return y + x0


class LightLocalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x0):
        x = self.norm(x0)
        x = self.dw(x)
        x = F.gelu(x)
        x = self.pw(x)
        return x + x0


class ComplexityRouter(nn.Module):
    def __init__(self, dim, wsize=16, reduction=4, init_bias=-1.0, gate_min=0.1, gate_max=0.9):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.wsize = wsize
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1)
        )
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        gate = torch.sigmoid(self.net(x))
        gate = self.gate_min + (self.gate_max - self.gate_min) * gate

        pad_h = (self.wsize - h % self.wsize) % self.wsize
        pad_w = (self.wsize - w % self.wsize) % self.wsize
        if pad_h > 0 or pad_w > 0:
            gate = F.pad(gate, (0, pad_w, 0, pad_h), 'reflect')

        gate = F.avg_pool2d(gate, kernel_size=self.wsize, stride=self.wsize)
        gate = gate.repeat_interleave(self.wsize, dim=2).repeat_interleave(self.wsize, dim=3)
        return gate[:, :, :h, :w]


class SoftComplexityAdaptiveBlock(nn.Module):
    def __init__(self, dim, heads=3, wsize=16, shift=False):
        super().__init__()
        assert heads == 3, "heads should be fixed to 3 in current setting"
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
        self.light = LightLocalBlock(dim)
        self.heavy = WindowAttention(dim, heads=heads, wsize=wsize, shift=shift)
        self.router = ComplexityRouter(dim, wsize=wsize, reduction=4, init_bias=-1.0, gate_min=.1, gate_max=0.9)

    def forward(self, x):
        light = self.light(x)
        heavy = self.heavy(x)
        gate = self.router(x)
        return light + gate * (heavy - light)


class MLP(nn.Module):
    def __init__(self, dim, ratio=1.8):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        expand = int(dim * ratio)
        if expand % 2 != 0:
            expand += 1
        self.proj1 = nn.Conv2d(dim, expand, 1)
        self.dwconv = nn.Conv2d(expand, expand, 3, 1, 1, groups=expand)
        self.proj2 = nn.Conv2d(expand // 2, dim, 1)

    def forward(self, x0):
        x = self.norm(x0)
        x = self.proj1(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.proj2(x)
        return x + x0


class BasicBlock(nn.Module):
    def __init__(self, dim, spatial=False, wsize=16, shift=False, is_dilated=False):
        super().__init__()
        self.spatial = spatial
        self.is_dilated = is_dilated
        if spatial:
            self.op = SoftComplexityAdaptiveBlock(dim, heads=3, wsize=wsize, shift=shift)
        else:
            self.op = LearnedTransformBlock(dim, base_window=8, rank=1)
        self.mlp = MLP(dim)

    def forward(self, x0):
        if not self.spatial or not self.is_dilated:
            x = self.op(x0)
            return self.mlp(x)

        x00, x01 = x0[:, :, 0::2, 0::2], x0[:, :, 0::2, 1::2]
        x10, x11 = x0[:, :, 1::2, 0::2], x0[:, :, 1::2, 1::2]

        y00, y01 = self.op(x00), self.op(x01)
        y10, y11 = self.op(x10), self.op(x11)

        y = torch.zeros_like(x0)
        y[:, :, 0::2, 0::2], y[:, :, 0::2, 1::2] = y00, y01
        y[:, :, 1::2, 0::2], y[:, :, 1::2, 1::2] = y10, y11

        return self.mlp(y)


class ResidualGroup(nn.Module):
    def __init__(self, dim, num=4, wsize=16, group_idx=0, total_groups=5):
        super().__init__()
        is_dilated = False
        self.blocks = nn.ModuleList()
        for i in range(num):
            self.blocks.append(BasicBlock(
                dim,
                spatial=(i >= 2),
                wsize=wsize,
                shift=((i + 2) % 4 == 0),
                is_dilated=is_dilated
            ))
        self.conv = nn.Conv2d(dim, dim, 1)
        self.ca = effCA(dim)

    def forward(self, x0):
        x = x0
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x)
        x = self.ca(x)
        return x0 + x


class FreeTransformSR(nn.Module):
    def __init__(self, up_scale=2, dim=60, groups=5, num=4):
        super().__init__()
        self.init = nn.Conv2d(3, dim, 3, 1, 1)
        self.body = nn.ModuleList([
            ResidualGroup(dim, num=num, wsize=16, group_idx=g, total_groups=groups)
            for g in range(groups)
        ])
        self.final_refine = nn.Conv2d(dim, dim, 1)
        self.up = nn.Sequential(
            nn.Conv2d(dim, 3 * up_scale ** 2, 3, 1, 1),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, x0):
        x = self.init(x0)
        feat_init = x
        for group in self.body:
            x = group(x)
        x = x + feat_init * 0.1
        x = self.final_refine(x)
        return self.up(x) + F.interpolate(x0, scale_factor=self.up_scale, mode='bilinear', align_corners=False)

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                try:
                    if own_state[name].shape == param.shape:
                        own_state[name].copy_(param.data)
                except Exception:
                    pass


if __name__ == "__main__":
    net = FreeTransformSR(up_scale=2, dim=60, groups=5, num=4)
    total = sum(p.numel() for p in net.parameters())
    print(f'Params: {total / 1e3:.2f}K')

    if HAS_FVCORE:
        input_lr = torch.randn(1, 3, 640, 360)
        net.eval()
        flops_fv = FlopCountAnalysis(net, input_lr)
        print(f'FLOPs (fvcore): {flops_fv.total() / 1e9:.2f} G')