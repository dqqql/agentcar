# 端到端POI深度推荐模型（四阶段融合）
# 阶段1: 客观打分(预计算s_obj) → 阶段2: 双线性融合 → 阶段3: Bi-LSTM+距离注意力 → 阶段4: 融合输出
import torch
import torch.nn as nn

from backend.app.services.ranking.model.bilinear_fusion import BilinearFusionLayer
from backend.app.services.ranking.model.distance_attention import DistanceAwareAttention


class POIDeepRecommender(nn.Module):
    """
    四阶段深度POI推荐模型:
    1. 客观打分（外部预计算，作为s_obj输入）
    2. 自适应双线性交叉融合（可训练M矩阵，创新点1）
    3. Bi-LSTM + 距离感知滑动窗口注意力（创新点2）
    4. 三得分融合输出最终推荐分数
    """

    def __init__(self, num_tags: int, embed_dim: int = 64,
                 window_size: int = 10, rho: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim

        # 标签嵌入表（冷启动: 用户/POI向量 = 标签嵌入均值）
        self.tag_embedding = nn.Embedding(num_tags, embed_dim, padding_idx=0)

        # 阶段2: 双线性融合层
        self.bilinear_layer = BilinearFusionLayer(embed_dim, init_identity=True)

        # 阶段3: Bi-LSTM (输入D, 隐藏D/2, 双向→输出D)
        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.distance_attn = DistanceAwareAttention(embed_dim, window_size, rho)

        # 阶段4: 融合头 [s_obj, s_sub, s_seq] → 最终得分
        self.score_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    # ---- 嵌入获取（冷启动策略: 标签哈希→嵌入均值） ----

    def get_user_embedding(self, tag_indices: torch.Tensor) -> torch.Tensor:
        """单用户向量: 偏好标签嵌入均值 [D]"""
        if tag_indices.numel() == 0:
            return torch.zeros(self.embed_dim, device=tag_indices.device)
        return self.tag_embedding(tag_indices).mean(dim=0)

    def get_poi_embedding(self, tag_indices: torch.Tensor) -> torch.Tensor:
        """单POI向量: 标签嵌入均值 [D]"""
        if tag_indices.numel() == 0:
            return torch.zeros(self.embed_dim, device=tag_indices.device)
        return self.tag_embedding(tag_indices).mean(dim=0)

    def get_user_embedding_batch(self, tag_indices_list: list[torch.Tensor]) -> torch.Tensor:
        """批处理用户向量 [batch, D]"""
        vecs = torch.zeros(len(tag_indices_list), self.embed_dim)
        for i, idx in enumerate(tag_indices_list):
            if idx.numel() > 0:
                vecs[i] = self.tag_embedding(idx).mean(dim=0)
        return vecs

    def get_poi_embedding_batch(self, tag_indices_list: list[torch.Tensor]) -> torch.Tensor:
        """批处理POI向量 [batch, D]"""
        vecs = torch.zeros(len(tag_indices_list), self.embed_dim)
        for i, idx in enumerate(tag_indices_list):
            if idx.numel() > 0:
                vecs[i] = self.tag_embedding(idx).mean(dim=0)
        return vecs

    # ---- 前向传播 ----

    def forward(self, user_vec, poi_vec, s_obj,
                history_seqs=None, history_locs=None, current_locs=None):
        """
        批处理前向传播
        user_vec:  [batch, D]
        poi_vec:   [batch, D]
        s_obj:     [batch, 1]
        history_seqs:  None 或 List[Tensor[seq_len, D]] (变长历史嵌入序列)
        history_locs:  None 或 List[Tensor[seq_len, 2]]
        current_locs:  None 或 [batch, 2]
        返回: s_total [batch,1], bilinear_feat [batch,1], s_seq [batch,1]
        """
        batch_size = user_vec.size(0)

        # 阶段2: 双线性融合
        _, bilinear_feat = self.bilinear_layer(user_vec, poi_vec, s_obj)

        # 阶段3: 序列预测（有历史轨迹时）
        s_seq = torch.zeros(batch_size, 1, device=user_vec.device)

        if history_seqs is not None:
            for i in range(batch_size):
                seq = history_seqs[i]
                if seq is not None and seq.size(0) > 0:
                    # Bi-LSTM编码
                    lstm_out, _ = self.lstm(seq.unsqueeze(0))  # [1, seq_len, D]
                    h_t = lstm_out[0, -1, :]  # [D] 最后时刻隐状态

                    # 距离感知注意力
                    context_vec, _ = self.distance_attn(
                        h_t,
                        lstm_out[0],        # [seq_len, D]
                        history_locs[i],    # [seq_len, 2]
                        current_locs[i],    # [2]
                    )
                    s_seq[i] = torch.sigmoid(context_vec.mean())

        # 阶段4: 融合
        features = torch.cat([s_obj, bilinear_feat, s_seq], dim=1)  # [batch, 3]
        s_total = self.score_head(features)  # [batch, 1]
        return s_total, bilinear_feat, s_seq

    @torch.no_grad()
    def inference_single(self, user_vec, poi_vec, s_obj,
                         history_seq=None, history_locs=None, current_loc=None):
        """线上推理: 单样本"""
        u = user_vec.unsqueeze(0)
        p = poi_vec.unsqueeze(0)
        s = s_obj.reshape(1, 1)
        hs = [history_seq] if history_seq is not None else None
        hl = [history_locs] if history_locs is not None else None
        cl = current_loc.unsqueeze(0) if current_loc is not None else None
        s_total, bilinear_feat, s_seq = self.forward(u, p, s, hs, hl, cl)
        return s_total.item(), bilinear_feat.item(), s_seq.item()
