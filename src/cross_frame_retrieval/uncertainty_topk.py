import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFrameAbsoluteAttn(nn.Module):
    """
    一个“绝对注意力”的跨帧模块：
      - 不做行归一化，而是对相似度 - 阈值做 ReLU
      - 若所有相似度都 < 阈值，则行全 0 (表示无匹配)
      - 阈值 t_{h,i} 由网络从 Q 中自适应学到
      - 多头机制可让网络并行学习不同的关注模式
    """
    def __init__(
        self,
        in_channels=3,    # 输入帧 (RGB) 通道数
        feat_dim=4,       # VAE 特征通道数
        embed_dim=32,     # Query/Key/Value 的内部维度(单头)
        heads=9
    ):
        super().__init__()
        self.heads = heads
        self.embed_dim = embed_dim

        # Q, K 的通道 = heads * embed_dim
        self.query_conv = nn.Conv2d(in_channels,  embed_dim*heads, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels,  embed_dim*heads, kernel_size=1)
        # V 的通道 = heads * embed_dim
        self.value_conv = nn.Conv2d(feat_dim, embed_dim*heads, kernel_size=1)

        # 输出投影
        self.proj_out = nn.Conv2d(embed_dim*heads, feat_dim, kernel_size=1)
        # self.temperature = nn.Parameter(torch.tensor(3000.0))
        self.temperature = torch.tensor(3000.0)

        # 阈值投影：把每个 head 的 Q 向量映射成 1 个标量阈值
        #   这里简单用一个 linear: [head_dim] -> [1]
        #   但我们得先 flatten/spatial reshape，再点乘
        self.threshold_fc = nn.Linear(embed_dim, 1)

    def forward(
        self,
        second_frame: torch.Tensor,            # [B,3,H,W]
        first_frame_aligned: torch.Tensor,     # [B,3,H,W]
        second_frame_feat: torch.Tensor,       # [B,feat_dim,H//8,W//8]
        first_frame_feat_aligned: torch.Tensor # [B,feat_dim,H//8,W//8]
    ):
        B, _, H, W = second_frame.shape
        HW = H * W

        # 为了对齐在同一空间尺度上做像素级 attention，这里把 VAE 特征上采样
        # 注意：如果想在1/8尺度上做，也可把 (Q,K) 下采样到同尺度即可。
        first_frame_feat_up = F.interpolate(
            first_frame_feat_aligned, size=(H, W),
            mode='bilinear', align_corners=False
        )  # [B, feat_dim, H, W]

        # 1) 生成 Q, K, V
        Q = self.query_conv(second_frame)        # [B, heads*embed_dim, H, W]
        K = self.key_conv(first_frame_aligned)   # [B, heads*embed_dim, H, W]
        V = self.value_conv(first_frame_feat_up) # [B, heads*embed_dim, H, W]

        # 2) reshape 成多头 [B, heads, embed_dim, H*W]
        Q = Q.view(B, self.heads, self.embed_dim, HW)
        K = K.view(B, self.heads, self.embed_dim, HW)
        V = V.view(B, self.heads, self.embed_dim, HW)

        # 3) 计算相似度 sim(i,j) = Q[i] dot K[j], 不做softmax
        #    sim: [B, heads, HW, HW]
        #    Q: [B, heads, embed_dim, HW] -> permute to [B, heads, HW, embed_dim]
        Q_ = Q.permute(0, 1, 3, 2)   # [B, heads, HW, embed_dim]
        K_ = K.permute(0, 1, 2, 3)   # [B, heads, embed_dim, HW]
        d = self.embed_dim
        sim = torch.matmul(Q_, K_) / math.sqrt(d)  # [B, heads, HW, HW]

        # print("sim.min: {}, sim.max: {}".format(sim.min(), sim.max()))

        # 4) 可学习阈值: threshold[i] = FC(Q[i])
        #    先把 Q_ reshape 为 [B*heads*HW, embed_dim]
        #    => fc => [B*heads*HW, 1]
        #    => reshape back to [B, heads, HW]
        Q_for_thresh = Q_.reshape(B*self.heads*HW, d)  # [B*heads*HW, embed_dim]
        t = self.threshold_fc(Q_for_thresh)            # [B*heads*HW,1]
        t = t.view(B, self.heads, HW)                  # [B,heads,HW]
        # print("t.min: {}, t.max: {}".format(t.min(), t.max()))

        # 我们要让 sim(i,j) 跟 t(i) 做比较：
        # attn(i,j) = relu(sim(i,j) - t(i))
        # => attn: [B, heads, HW, HW]
        # 先扩展 t 到 [B,heads,HW,1], 与 sim 的最后一维 HW 对齐
        t_expanded = t.unsqueeze(-1)  # [B,heads,HW,1]
        attn = sim - t_expanded       # [B,heads,HW,HW]
        attn = F.relu(attn/self.temperature)
        # attn = F.softmax(attn, dim=-1)          # 小于0则变0

        # print("attn.min: {}, max: {}".format(attn.min(), attn.max()))

        # 5) 用 attn 加权 V
        #    V_ = [B, heads, HW, embed_dim]
        V_ = V.permute(0, 1, 3, 2)  # => [B, heads, HW, embed_dim]
        # print("V.min: {}, V.max: {}".format(V_.min(), V_.max()))
        out_attn = torch.matmul(attn, V_)  # [B, heads, HW, embed_dim]
        # print("out_attn.min: {}, attn.max: {}".format(out_attn.min(), out_attn.max()))
        # out_attn: [B, heads, HW, embed_dim]
        # 再转成 [B, heads*embed_dim, H, W]
        out_attn = out_attn.permute(0, 1, 3, 2)  # => [B, heads, embed_dim, HW]
        out_attn = out_attn.reshape(B, self.heads*self.embed_dim, H, W)

        # print("out_attn.min: {}, attn.max: {}".format(out_attn.min(), out_attn.max()))

        # 6) proj_out => [B, feat_dim, H, W]
        out_feat = self.proj_out(out_attn)

        # 7) 若要再还原到 [B,feat_dim,H//8,W//8] 与 second_frame_feat 对齐:
        out_feat_down = F.interpolate(
            out_feat,
            size=second_frame_feat.shape[-2:],
            mode='bilinear', align_corners=False
        )
        # print("first_frame_img: {}, first_frame_img: {}".format(first_frame_aligned.min(), first_frame_aligned.max()))
        # print("first_frame_feat_up: {}, first_frame_feat_up: {}".format(first_frame_feat_up.min(), first_frame_feat_up.max()))
        # print("first_frame: {}, first_frame: {}".format(out_feat_down.min(), out_feat_down.max()))
        # print("second_frame: {}, second_frame: {}".format(second_frame.min(), second_frame.max()))
        # print("----------second_frame_feat.min: {}, max: {}".format(second_frame_feat.min(), second_frame_feat.max()))
        final_feat = second_frame_feat + out_feat_down

        # 返回得到的跨帧特征 以及 注意力图 attn
        # attn 形状 [B,heads,HW,HW], 行并不归一化,
        # 若想可视化成图，需要再 reshape (HW->H,W)，对每个head单独查看
        return final_feat, out_feat_down
