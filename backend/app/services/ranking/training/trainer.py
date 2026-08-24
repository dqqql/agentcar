# 训练器：训练循环 + 早停 + Checkpoint管理 + 指标日志
import os
import json
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.app.services.ranking.model.poi_recommender import POIDeepRecommender
from backend.app.services.ranking.training.losses import compute_total_loss, compute_ndcg_at_k


class Trainer:
    def __init__(self, model: POIDeepRecommender, config: dict,
                 tag_vocab: dict, poi_list: list, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.tag_vocab = tag_vocab
        self.poi_list = poi_list

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["eta"]
        )
        self.best_ndcg = 0.0
        self.patience_counter = 0
        self.history: list[dict] = []

    def _make_batch_tensors(self, batch_interactions, all_interactions):
        """将batch转换为模型输入张量"""
        bsz = len(batch_interactions)
        user_vecs = torch.zeros(bsz, self.model.embed_dim, device=self.device)
        poi_vecs = torch.zeros(bsz, self.model.embed_dim, device=self.device)
        s_objs = torch.zeros(bsz, 1, device=self.device)
        labels = torch.zeros(bsz, 1, device=self.device)

        z_str_list = [None] * bsz
        z_wk_list = [None] * bsz
        z_neg_list = [None] * bsz

        for i, inter in enumerate(batch_interactions):
            # 用户/POI嵌入
            u_idx = torch.tensor(inter.user_tag_indices, dtype=torch.long, device=self.device)
            p_idx = torch.tensor(inter.poi_tag_indices, dtype=torch.long, device=self.device)
            user_vecs[i] = self.model.get_user_embedding(u_idx)
            poi_vecs[i] = self.model.get_poi_embedding(p_idx)
            s_objs[i] = inter.s_obj
            labels[i] = inter.label

            # 对比学习样本
            if inter.strong_pos_idx is not None and inter.strong_pos_idx < len(all_interactions):
                sp = all_interactions[inter.strong_pos_idx]
                sp_pidx = torch.tensor(sp.poi_tag_indices, dtype=torch.long, device=self.device)
                z_str_list[i] = self.model.get_poi_embedding(sp_pidx)

            if inter.weak_pos_idx is not None and inter.weak_pos_idx < len(all_interactions):
                wp = all_interactions[inter.weak_pos_idx]
                wp_pidx = torch.tensor(wp.poi_tag_indices, dtype=torch.long, device=self.device)
                z_wk_list[i] = self.model.get_poi_embedding(wp_pidx)

                # 负样本: 随机采样5个label=0的interaction
                neg_indices = []
                attempts = 0
                while len(neg_indices) < 5 and attempts < 50:
                    j = torch.randint(0, len(all_interactions), (1,)).item()
                    if all_interactions[j].label == 0:
                        np_pidx = torch.tensor(
                            all_interactions[j].poi_tag_indices, dtype=torch.long, device=self.device)
                        neg_indices.append(self.model.get_poi_embedding(np_pidx))
                    attempts += 1
                z_neg_list[i] = neg_indices if neg_indices else None

        return user_vecs, poi_vecs, s_objs, labels, z_str_list, z_wk_list, z_neg_list

    def train_epoch(self, train_data: list, all_interactions: list):
        self.model.train()
        total_loss = 0.0
        total_metrics = {"L_rec": 0, "L_cl": 0, "L2": 0, "total": 0}
        n_batches = 0

        batch_size = self.config["batch_size"]
        indices = torch.randperm(len(train_data)).tolist()

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [train_data[i] for i in batch_idx]

            user_vecs, poi_vecs, s_objs, labels, z_str, z_wk, z_neg = \
                self._make_batch_tensors(batch, all_interactions)

            # 前向传播 (不使用历史序列，纯双线性+融合)
            s_total, bilinear_feats, s_seq = self.model(
                user_vecs, poi_vecs, s_objs, history_seqs=None
            )

            # 计算损失 (使用POI嵌入作为对比学习表示)
            loss, metrics = compute_total_loss(
                s_total, labels, poi_vecs, self.model,
                z_str=z_str, z_wk=z_wk, z_neg_list=z_neg,
                tau=self.config["tau"], gamma=self.config["gamma"],
                lambda_cl=self.config["lambda_cl"], eta=self.config["eta"],
                sigma=self.config["sigma"], mu=self.config["mu"],
            )

            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 差分隐私梯度加噪 (sigma>0时)
            if self.config["sigma"] > 0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, self.config["sigma"], size=param.grad.shape,
                                            device=param.grad.device)
                        param.grad = param.grad + noise

            self.optimizer.step()

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0)
            n_batches += 1

        return total_loss / max(n_batches, 1), {
            k: v / max(n_batches, 1) for k, v in total_metrics.items()
        }

    @torch.no_grad()
    def validate(self, val_data: list, all_interactions: list):
        """验证: 计算NDCG@10(按用户分组)和平均损失"""
        self.model.eval()
        total_loss = 0.0
        # 按用户收集预测分和真实标签
        user_scores: dict[int, list[float]] = {}
        user_labels: dict[int, list[float]] = {}

        batch_size = self.config["batch_size"]
        for start in range(0, len(val_data), batch_size):
            batch = val_data[start:start + batch_size]
            user_vecs, poi_vecs, s_objs, labels, z_str, z_wk, z_neg = \
                self._make_batch_tensors(batch, all_interactions)

            s_total, bilinear_feats, _ = self.model(
                user_vecs, poi_vecs, s_objs, history_seqs=None
            )

            loss, _ = compute_total_loss(
                s_total, labels, poi_vecs, self.model,
                z_str=z_str, z_wk=z_wk, z_neg_list=z_neg,
                **{k: self.config[k] for k in ["tau", "gamma", "lambda_cl", "eta", "sigma", "mu"]}
            )
            total_loss += loss.item()

            # 收集每个用户的预测分和标签
            for i, inter in enumerate(batch):
                uid = inter.user_id
                user_scores.setdefault(uid, []).append(s_total[i].item())
                user_labels.setdefault(uid, []).append(labels[i].item())

        # 按用户计算NDCG@10
        ndcg_scores = []
        for uid in user_scores:
            scores_t = torch.tensor(user_scores[uid])
            labels_t = torch.tensor(user_labels[uid])
            ndcg = compute_ndcg_at_k(scores_t, labels_t, k=10)
            ndcg_scores.append(ndcg)

        avg_ndcg = sum(ndcg_scores) / max(len(ndcg_scores), 1)
        return total_loss / max(len(val_data) // batch_size + 1, 1), avg_ndcg

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        version = self.config.get("version", "v0.1.0")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{version}_ep{epoch:04d}_ndcg{metrics['val_ndcg']:.4f}_{ts}.pt"
        path = os.path.join(ckpt_dir, name)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config,
            "tag_vocab": self.tag_vocab,
            "num_tags": len(self.tag_vocab),
            "embed_dim": self.model.embed_dim,
            "poi_list": self.poi_list,
            "saved_at": datetime.now().isoformat(),
        }, path)

        if is_best:
            best_path = os.path.join(ckpt_dir, "best.pt")
            shutil.copy2(path, best_path)
            print(f"  [BEST] 新最佳模型已保存: {best_path}")
        print(f"  Checkpoint已保存: {path}")
        return path

    def train(self, train_data: list, val_data: list, all_interactions: list,
              ckpt_dir: str) -> list[dict]:
        patience = self.config.get("early_stop_patience", 3)
        epochs = self.config["epochs"]

        print(f"\n{'='*60}")
        print(f"开始训练 | 设备: {self.device} | 总样本: {len(train_data)} | 验证: {len(val_data)}")
        print(f"Epochs: {epochs} | Batch: {self.config['batch_size']} | LR: {self.config['lr']}")
        print(f"embed_dim: {self.model.embed_dim} | sigma: {self.config['sigma']}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_metrics = self.train_epoch(train_data, all_interactions)

            # 验证
            val_loss, val_ndcg = self.validate(val_data, all_interactions)

            epoch_log = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_L_rec": train_metrics.get("L_rec", 0),
                "train_L_cl": train_metrics.get("L_cl", 0),
                "val_loss": round(val_loss, 6),
                "val_ndcg": round(val_ndcg, 4),
            }
            self.history.append(epoch_log)

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} (rec={train_metrics.get('L_rec', 0):.4f} "
                  f"cl={train_metrics.get('L_cl', 0):.4f}) | "
                  f"Val Loss: {val_loss:.6f} | NDCG@10: {val_ndcg:.4f}")

            # 早停 + Checkpoint
            is_best = val_ndcg > self.best_ndcg
            if is_best:
                self.best_ndcg = val_ndcg
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, epoch_log, is_best, ckpt_dir)

            if self.patience_counter >= patience:
                print(f"\n早停触发: {patience}轮未提升 (best NDCG={self.best_ndcg:.4f})")
                break

        print(f"\n训练完成! 最佳NDCG@10: {self.best_ndcg:.4f}")
        return self.history
