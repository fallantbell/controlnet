from inspect import isfunction
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class Epipolar_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, do_epipolar,do_bidirectional_epipolar,do_blur, qkv_bias = True, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.do_epipolar = do_epipolar
        self.do_bidirectional_epipolar = do_bidirectional_epipolar
        self.do_blur = do_blur

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
    
    def get_epipolar(self,b,h,w,k,src_c2w,target_c2w):
        H = h
        W = H*16/9  #* 原始圖像為 16:9

        k = k.to(dtype=torch.float32)
        src_c2w=src_c2w.to(dtype=torch.float32)
        target_c2w=target_c2w.to(dtype=torch.float32)

        #* unormalize intrinsic 

        k[:,0] = k[:,0]*W
        k[:,1] = k[:,1]*H

        k[:,0,2] = h/2
        k[:,1,2] = h/2

        device = k.device

        #* 創建 h*w 的 uv map
        x_coords, y_coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32)
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)

        x_coords = x_coords.to(device)
        y_coords = y_coords.to(device)
        coords_tensor = coords_tensor.to(device)

        k_3x3 = k[:,0:3,0:3]
        src_c2w_r = src_c2w[:,0:3,0:3]
        src_c2w_t = src_c2w[:,0:3,3]
        target_c2w_r = target_c2w[:,0:3,0:3]
        target_c2w_t = target_c2w[:,0:3,3]
        target_w2c_r = torch.linalg.inv(target_c2w_r)
        target_w2c_t = -target_c2w_t

        cx = k_3x3[:,0,2].view(b, 1)
        cy = k_3x3[:,1,2].view(b, 1)
        fx = k_3x3[:,0,0].view(b, 1)
        fy = k_3x3[:,1,1].view(b, 1)
        coords_tensor[...,0] = (coords_tensor[...,0]-cx)/fx
        coords_tensor[...,1] = (coords_tensor[...,1]-cy)/fy

        #* 做 H*W 個點的運算
        coords_tensor = rearrange(coords_tensor, 'b hw p -> b p hw') 
        point_3d_world = torch.matmul(src_c2w_r,coords_tensor)              #* 相機坐標系 -> 世界座標
        point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)           #* 相機坐標系 -> 世界座標
        point_2d = torch.matmul(target_w2c_r,point_3d_world)                #* 世界座標 -> 相機座標
        point_2d = point_2d + target_w2c_t.unsqueeze(-1)                    #* 世界座標 -> 相機座標
        pi_to_j = torch.matmul(k_3x3,point_2d)                              #* 相機座標 -> 平面座標

        #* 原點的計算
        oi = torch.zeros(3).to(dtype=torch.float32)
        oi = repeat(oi, 'p -> b p', b=b)
        oi = oi.unsqueeze(-1)
        oi = oi.to(device)
        point_3d_world = torch.matmul(src_c2w_r,oi)
        point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)  
        point_2d = torch.matmul(target_w2c_r,point_3d_world)
        point_2d = point_2d + target_w2c_t.unsqueeze(-1)  
        oi_to_j = torch.matmul(k_3x3,point_2d)
        oi_to_j = rearrange(oi_to_j, 'b c p -> b p c') #* (b,3,1) -> (b,1,3)

        #* 除以深度
        pi_to_j_unnormalize = rearrange(pi_to_j, 'b p hw -> b hw p') 
        pi_to_j = pi_to_j_unnormalize / (pi_to_j_unnormalize[..., -1:] + 1e-6)   #* (b,hw,3)
        # pi_to_j = pi_to_j_unnormalize / pi_to_j_unnormalize[..., -1:]
        oi_to_j = oi_to_j / oi_to_j[..., -1:]   #* (b,1,3)

        # print(f"pi_to_j: {pi_to_j[0,9]}")
        # print(f"oi_to_j: {oi_to_j[0,0]}")

        #* 計算feature map 每個點到每個 epipolar line 的距離
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32) # (4096,3)
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)
        coords_tensor = coords_tensor.to(device)

        oi_to_pi = pi_to_j - oi_to_j            #* h*w 個 epipolar line (b,hw,3)
        oi_to_coord = coords_tensor - oi_to_j   #* h*w 個點   (b,hw,3)

        ''''
            #* 這裡做擴展
            oi_to_pi    [0,0,0]     ->      oi_to_pi_repeat     [0,0,0]
                        [1,1,1]                                 [0,0,0]
                        [2,2,2]                                 [1,1,1]
                                                                [1,1,1]
                                                                .
                                                                .
                                                                .

            oi_to_coord     [0,0,0]     ->      oi_to_coord_repeat      [0,0,0]
                            [1,1,1]                                     [1,1,1]
                            [2,2,2]                                     [2,2,2]
                                                                        [0,0,0]
                                                                        .
                                                                        .
                                                                        .
        '''
        oi_to_pi_repeat = repeat(oi_to_pi, 'b i j -> b i (repeat j)',repeat = h*w)
        oi_to_pi_repeat = rearrange(oi_to_pi_repeat,"b i (repeat j) -> b (i repeat) j", repeat = h*w)
        oi_to_coord_repeat = repeat(oi_to_coord, 'b i j -> b (repeat i) j',repeat = h*w)


        area = torch.cross(oi_to_pi_repeat,oi_to_coord_repeat,dim=-1)     #* (b,hw*hw,3)
        area = torch.norm(area,dim=-1 ,p=2)
        vector_len = torch.norm(oi_to_pi_repeat, dim=-1, p=2)
        distance = area/vector_len

        distance_weight = 1 - torch.sigmoid(50*(distance-0.5))

        epipolar_map = rearrange(distance_weight,"b (hw hw2) -> b hw hw2",hw = h*w)

        #* 如果 max(1-sigmoid) < 0.5 
        #* => min(distance) > 0.05 
        #* => 每個點離epipolar line 太遠
        #* => epipolar line 不在圖中
        #* weight map 全設為 1 
        max_values, _ = torch.max(epipolar_map, dim=-1)
        mask = max_values < 0.5
        epipolar_map[mask.unsqueeze(-1).expand_as(epipolar_map)] = 1

        if (torch.any(torch.isnan(epipolar_map)) or
            torch.any(torch.isnan(distance)) or
            torch.any(torch.isnan(distance_weight)) or
            torch.any(torch.isnan(area)) or
            torch.any(torch.isnan(vector_len)) or        
            torch.any(torch.isnan(distance_weight)) or
            torch.any(torch.isnan(oi_to_pi_repeat)) or
            torch.any(torch.isnan(oi_to_coord_repeat))):
            print(f"find nan !!!")
            print(f"epipolar_map: {torch.any(torch.isnan(epipolar_map))}")
            print(f"distance_weight: {torch.any(torch.isnan(distance_weight)) }")
            print(f"distance: {torch.any(torch.isnan(distance)) }")
            print(f"vector_len: {torch.any(torch.isnan(vector_len)) }")
            print(f"area: {torch.any(torch.isnan(area)) }")
            print(f"oi_to_pi_repeat: {torch.any(torch.isnan(oi_to_pi_repeat))}")
            print(f"oi_to_coord_repeat: {torch.any(torch.isnan(oi_to_coord_repeat))}")
            print(f"pi_to_j: {torch.any(torch.isnan(pi_to_j))}")
            print(f"oi_to_j: {torch.any(torch.isnan(oi_to_j))}")
            print(f"pi_to_j_unnormalize has zero: {torch.any(torch.eq(pi_to_j_unnormalize[...,-1:],0))}")
            print(" ")
            print("break !")
            os._exit(0)



        return epipolar_map


    def forward(self, x, src_encode,intrinsic = None,c2w = None):
        b, c, h, w = x.shape

        """
        q: (b d H W)
        k: (b d h w)
        v: (b d h w)
        """
        _, _, H, W = x.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(x, 'b d H W -> b (H W) d')
        k = rearrange(src_encode, 'b d h w -> b (h w) d')
        v = rearrange(src_encode, 'b d h w -> b (h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        #* 一般的 cross attention -> 得到 attention map
        cross_attend = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)

        weight_map = torch.ones_like(cross_attend)

        if self.do_epipolar:
            #* 得到 epipolar weighted map (B,HW,HW)
            epipolar_map = self.get_epipolar(b,h,w,intrinsic.clone(),c2w[1],c2w[0])

            epipolar_map = repeat(epipolar_map,'b hw hw2 -> (b repeat) hw hw2',repeat = self.heads)

            weight_map = weight_map*epipolar_map
        
        if self.do_bidirectional_epipolar:
            #* 做反方向的epipolar
            epipolar_map = self.get_epipolar(b,h,w,intrinsic.clone(),c2w[0],c2w[1])

            epipolar_map_transpose = epipolar_map.permute(0,2,1)

            epipolar_map = repeat(epipolar_map_transpose,'b hw hw2 -> (b repeat) hw hw2',repeat = self.heads)

            weight_map = weight_map*epipolar_map

            if self.do_blur:
                weight_map = rearrange(weight_map,'b hw hw2 -> (b hw) c height weight',c=1,height=h)
                if h == 64:
                    kernel_size = 7
                elif h == 32:
                    kernel_size = 5
                elif h == 16:
                    kernel_size = 3

                transform1 = T.GaussianBlur(kernel_size,1.5)
                blurred_weightmap = transform1(weight_map)
                weight_map = rearrange(blurred_weightmap,'(b hw) c h w -> b hw (h w)',hw=h*w,c=1)

        cross_attend = cross_attend * weight_map
        att = cross_attend.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z.contiguous()