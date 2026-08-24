# 阶段3：距离感知滑动窗口注意力（创新点2）
# 在注意力中植入空间距离衰减因子: exp(-rho * dist)
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.services.ranking.model.layers import haversine_distance_batch


class DistanceAwareAttention(nn.Module):
    """
    创新点2: 注意力权重 = softmax(h_t^T W_a h_j × exp(-rho × dist))
    - 滑动窗口截取最近w个历史POI
    - 距离衰减确保空间连续性（近邻历史权重更高）
    """

    def __init__(self, embed_dim: int, window_size: int = 10, rho: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.rho = rho
        self.W_a = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, h_t, history_hidden, history_locs, current_loc):
        """
        单样本前向（批处理时按样本循环调用）
        h_t:            [embed_dim] 当前Bi-LSTM隐状态
        history_hidden: [seq_len, embed_dim] 历史隐状态
        history_locs:   [seq_len, 2] 历史POI坐标(lat, lng)
        current_loc:    [2] 当前POI坐标
        返回: context_vec [embed_dim], attn_weights [w]
        """
        seq_len = history_hidden.size(0)
        if seq_len == 0:
            return torch.zeros_like(h_t), torch.tensor([], device=h_t.device)

        # 滑动窗口截取
        start = max(0, seq_len - self.window_size)
        window_h = history_hidden[start:]      # [w, D]
        window_locs = history_locs[start:]      # [w, 2]
        w = window_h.size(0)

        # 标准注意力: h_t^T W_a h_j
        projected = self.W_a(window_h)          # [w, D]
        scores = (h_t.unsqueeze(0) @ projected.T).squeeze(0)  # [w]

        # 空间距离衰减: exp(-rho * dist)
        curr = current_loc.unsqueeze(0).expand(w, -1)  # [w, 2]
        dists = haversine_distance_batch(curr, window_locs)  # [w] km
        spatial_penalty = torch.exp(-self.rho * dists)       # [w]

        # 融合语义相似度与空间衰减
        raw_weights = scores * spatial_penalty  # [w]
        normalized_weights = F.softmax(raw_weights, dim=0)  # [w]

        # 聚合上下文向量 c_t = Σ β_j h_j
        context_vector = (normalized_weights.unsqueeze(1) * window_h).sum(dim=0)  # [D]

        return context_vector, normalized_weights
