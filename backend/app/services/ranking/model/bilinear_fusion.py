# 阶段2：自适应双线性交叉融合层（创新点1）
# 将线性加权升级为二阶非线性协同: S_sub = U^T M V_i
import torch
import torch.nn as nn


class BilinearFusionLayer(nn.Module):
    """
    创新点1: 可学习的双线性映射矩阵M捕获用户-POI二阶特征交互
    - M初始化为单位矩阵: 训练初期退化为线性点积，保证收敛稳定
    - 训练后M可挖掘跨维度特征兼容性模式
    """

    def __init__(self, embed_dim: int, init_identity: bool = True):
        super().__init__()
        # 可学习双线性映射矩阵 M ∈ R^{D×D}
        self.M = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        if init_identity:
            with torch.no_grad():
                self.M.copy_(torch.eye(embed_dim) + torch.randn(embed_dim, embed_dim) * 0.01)

        # 融合层: [s_obj, s_sub] → 标量
        self.W_f = nn.Linear(2, 1)

    def forward(self, user_vec: torch.Tensor, poi_vec: torch.Tensor, s_obj: torch.Tensor):
        """
        批处理前向传播
        user_vec:  [batch, embed_dim]
        poi_vec:   [batch, embed_dim]
        s_obj:     [batch, 1] 客观得分
        返回: s_total [batch, 1], bilinear_feat [batch, 1]
        """
        # 双线性交互: (U @ M) * V → sum over dim → [batch, 1]
        bilinear_interaction = ((user_vec @ self.M) * poi_vec).sum(dim=1, keepdim=True)

        # 拼接客观分和主观分
        concat_feat = torch.cat([s_obj, bilinear_interaction], dim=1)  # [batch, 2]
        fusion_out = self.W_f(concat_feat)  # [batch, 1]

        s_total = torch.sigmoid(fusion_out)
        return s_total, bilinear_interaction
