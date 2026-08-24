# 损失函数：对比损失 + 差分隐私正则（含量纲校准）
import math
import torch
import torch.nn.functional as F


def compute_contrastive_loss(z, z_str, z_wk, z_neg_list, tau=0.07, gamma=0.5):
    """
    公式(6): 强弱正样本区分的对比损失 (InfoNCE变体)
    z:       [D] 锚点特征（POI嵌入向量）
    z_str:   [D] 强正样本（同轨迹下一POI嵌入）
    z_wk:    [D] 弱正样本（同用户相似POI嵌入）
    z_neg:   list[Tensor[D]] 负样本嵌入
    """
    sim_str = F.cosine_similarity(z.unsqueeze(0), z_str.unsqueeze(0)).squeeze() / tau
    sim_wk = F.cosine_similarity(z.unsqueeze(0), z_wk.unsqueeze(0)).squeeze() / tau

    numerator = torch.exp(sim_str) + gamma * torch.exp(sim_wk)
    denominator = numerator.clone()
    for z_neg in z_neg_list:
        sim_neg = F.cosine_similarity(z.unsqueeze(0), z_neg.unsqueeze(0)).squeeze() / tau
        denominator = denominator + torch.exp(sim_neg)

    loss_cl = -torch.log(numerator / (denominator + 1e-8))
    return loss_cl


def compute_total_loss(s_total, labels, representations, model,
                       z_str=None, z_wk=None, z_neg_list=None,
                       tau=0.07, gamma=0.5, lambda_cl=0.1, eta=1e-5,
                       sigma=0.0, mu=0.001):
    """
    公式(7): 差分隐私对比联合优化总损失
    representations: [batch, D] POI嵌入向量（用于对比学习）
    返回: (total_loss, metrics_dict)
    """
    # 1. 推荐损失 L_rec (BCE)
    L_rec = F.binary_cross_entropy(s_total.clamp(1e-7, 1 - 1e-7), labels, reduction="mean")

    # 2. 对比学习损失 L_cl (含量纲校准: InfoNCE值域约[0,10]，压缩到[0,0.4])
    L_cl = torch.tensor(0.0, device=s_total.device)
    n_cl = 0
    if z_str is not None and z_wk is not None and z_neg_list is not None:
        for i in range(s_total.size(0)):
            if i < len(z_str) and z_str[i] is not None and z_wk[i] is not None:
                negs = [z_neg_list[i][j] for j in range(len(z_neg_list[i]))] if z_neg_list[i] else []
                if negs:
                    # 使用POI嵌入向量作为对比学习的表示
                    cl = compute_contrastive_loss(
                        representations[i], z_str[i], z_wk[i], negs, tau, gamma)
                    L_cl = L_cl + cl
                    n_cl += 1
        if n_cl > 0:
            L_cl = L_cl / n_cl

    # 3. L2正则化 (量纲校准: 乘以eta压到极小)
    L2_reg = sum(p.norm(2) for p in model.parameters())

    # 4. 总损失 (差分隐私σ=0时关闭隐私项)
    total = L_rec + lambda_cl * L_cl + eta * L2_reg
    if sigma > 0:
        grad_penalty = mu * sum(
            p.grad.norm(2) for p in model.parameters()
            if p.grad is not None
        )
        total = total + grad_penalty

    metrics = {
        "L_rec": round(L_rec.item(), 6),
        "L_cl": round(L_cl.item() if isinstance(L_cl, torch.Tensor) else 0.0, 6),
        "L2": round((eta * L2_reg).item(), 6),
        "total": round(total.item(), 6),
    }
    return total, metrics


def compute_ndcg_at_k(predicted_scores: torch.Tensor, true_labels: torch.Tensor, k: int = 10) -> float:
    """NDCG@K排序指标"""
    k = min(k, predicted_scores.numel())
    sorted_idx = torch.argsort(predicted_scores, descending=True)
    sorted_labels = true_labels[sorted_idx]

    dcg = sum((2 ** sorted_labels[i].item() - 1) / math.log2(i + 2) for i in range(k))

    ideal_sorted, _ = torch.sort(true_labels, descending=True)
    idcg = sum((2 ** ideal_sorted[i].item() - 1) / math.log2(i + 2) for i in range(k))

    return float(dcg / idcg) if idcg > 0 else 0.0
