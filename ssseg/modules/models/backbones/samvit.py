'''
Function:
    Implementation of SAMViT
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bricks import BuildActivation, BuildNormalization


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''MLPBlock'''
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, act_cfg={'type': 'GELU'}):
        super(MLPBlock, self).__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = BuildActivation(act_cfg=act_cfg)
    '''forward'''
    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


'''LayerNorm2d'''
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    '''forward'''
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


'''Attention'''
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False, rel_pos_zero_init=True, input_size=None):
        super(Attention, self).__init__()
        # set attributes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        # initialize relative positional embeddings
        if self.use_rel_pos:
            assert input_size is not None, "input_size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
    '''forward'''
    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = self.adddecomposedrelpos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x
    '''adddecomposedrelpos'''
    def adddecomposedrelpos(self, attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.getrelpos(q_h, k_h, rel_pos_h)
        Rw = self.getrelpos(q_w, k_w, rel_pos_w)
        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)
        return attn
    '''getrelpos'''
    def getrelpos(self, q_size, k_size, rel_pos):
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear")
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos
        # scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos_resized[relative_coords.long()]


'''Block'''
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_cfg={'type': 'LayerNorm', 'eps': 1e-6}, act_cfg={'type': 'GELU'}, use_rel_pos=False, 
                 rel_pos_zero_init=True, window_size=0, input_size=None):
        super(Block, self).__init__()
        self.norm1 = BuildNormalization(placeholder=dim, norm_cfg=norm_cfg)
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = BuildNormalization(placeholder=dim, norm_cfg=norm_cfg)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act_cfg=act_cfg)
        self.window_size = window_size
    '''forward'''
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self.windowpartition(x, self.window_size)
        x = self.attn(x)
        # reverse window partition
        if self.window_size > 0:
            x = self.windowunpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
    '''windowpartition'''
    def windowpartition(self, x, window_size):
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)
    '''windowunpartition'''
    def windowunpartition(self, windows, window_size, pad_hw, hw):
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


'''PatchEmbed'''
class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
    '''forward'''
    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


'''SAMViT'''
class SAMViT(nn.Module):
    def __init__(
        self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, out_chans=256, qkv_bias=True, norm_cfg={'type': 'LayerNorm', 'eps': 1e-6},
        act_cfg={'type': 'GELU'}, use_abs_pos=True, use_rel_pos=False, rel_pos_zero_init=True, window_size=0, global_attn_indexes=(),
    ):
        super(SAMViT, self).__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim,
        )
        self.pos_embed = None
        # initialize absolute positional embedding with pretrain image size.
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_cfg=norm_cfg, act_cfg=act_cfg, use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init, window_size=window_size if i not in global_attn_indexes else 0, input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )
    '''forward'''
    def forward(self, x, return_interm_embeddings=False):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        interm_embeddings = []
        for blk in self.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        if return_interm_embeddings:
            return x, interm_embeddings
        return x