
from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# 判断 val 是否是元组类型。，如果不是将其转换为depth个元组。(val,)表示是一个元组
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)


# 批归一化更适合全连接层和卷积层，而层归一化更适合循环神经网络和Transformer。
# 层归一化（Layer Normalization）：批归一化是对一个batch的样本进行归一化，而层归一化是对单个样本的特征进行归一化。
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        # 一个很小的常数，用于防止分母为零。
        self.eps = eps
        # g,b可学习的参数，分别用于缩放和偏移归一化后的值。
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # 计算tensor的均值和方差
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# PreNorm 类在 Transformer 模型中是一种常用的归一化方式。
# fn: 需要归一化的子模块，可以是多头注意力模块、前馈神经网络等。
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # 对输入 x 进行 LayerNorm 归一化。
        # 然后将归一化的结果作为输入，传递给子模块 fn。
        # **kwargs：用于传递给子模块 fn 的其他参数。
        return self.fn(self.norm(x), **kwargs)

# 定义了一个前馈神经网络模块,主要作用是在 Transformer 的编码器和解码器中引入非线性变换
class FeedForward(nn.Module):
    # mlp_mult: 中间层特征维度的倍数，用于控制网络的宽度。
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            # 将输入特征维度扩大 mlp_mult 倍，引入非线性变换。特征维度扩大，增加模型的表达能力。
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 实现了多头注意力机制（Multi-Head Attention）
class Attention(nn.Module):
    # heads: 多头注意力的头数
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        #  // 表示整数除法,用于计算两个整数的整数结果,舍弃小数部分。计算每个头的维度 dim_head 和内层维度 inner_dim。
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Softmax 层用于计算注意力权重。
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        # Conv2d 层用于将输入特征映射到查询、键、值矩阵。
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 将查询序列、键、值矩阵重组为 b h (x y) d 的形状，以便进行注意力计算。
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# 用于聚合特征的模块,提取局部特征，从而提取更高级的特征。
def Aggregate(dim, dim_out):
    return nn.Sequential(
        # 卷积核大小3*3
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        # 下采样，减少特征图的大小
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        # 创建一个空的 nn.ModuleList 存储 Transformer 层。
        # 创建一个位置编码向量 self.pos_emb，用于引入位置信息。
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        # 构建 Transformer 层，为每个 Transformer 层创建一个 nn.ModuleList，包含一个注意力模块和一个前馈神经网络模块。
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        # 截取位置编码向量，使其与输入张量的形状匹配。
        # 将位置编码向量添加到输入张量。
        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        # 遍历 Transformer 层，依次执行注意力模块和前馈神经网络模块。
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class HTNet(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2 #
        fmap_size = image_size // patch_size #
        blocks = 2 ** (num_hierarchies - 1)#

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        # 将图像分割成 patches，嵌入到特征空间: 将每个 patch 转换为一个固定维度的特征向量。
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        # 构建多层 Transformer 模块
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        # MLP 头部
        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        # 将图像转换为 patch 嵌入
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)
        return self.mlp_head(x)

# This function is to confuse three models                            ？？？？？是不是没有用到
class Fusionmodel(nn.Module):
  def __init__(self):
    #  extend from original
    super(Fusionmodel,self).__init__()
    self.fc1 = nn.Linear(15, 3)
    self.bn1 = nn.BatchNorm1d(3)
    self.d1 = nn.Dropout(p=0.5)
    self.fc_2 = nn.Linear(6, 3)
    self.relu = nn.ReLU()
    # forward layers is to use these layers above
  def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
    fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
    fuse_out = self.fc1(fuse_four_features)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out) # drop out
    fuse_whole_four_parts = torch.cat(
        (whole_feature,fuse_out), 0)
    fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
    fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
    out = self.fc_2(fuse_whole_four_parts)
    return out
