#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POI深度推荐模型训练脚本
用法: python scripts/train_ranking_model.py [--epochs 10] [--embed-dim 64] [--device cuda]
"""
import os
import sys
import json
import argparse
from datetime import datetime

# 确保项目根目录在sys.path中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

from backend.app.services.ranking.model.poi_recommender import POIDeepRecommender
from backend.app.services.ranking.training.synthetic_data import SyntheticDataGenerator
from backend.app.services.ranking.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="训练POI深度推荐模型")
    parser.add_argument("--data-root", default=PROJECT_ROOT, help="项目根目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-3, help="学习率")
    parser.add_argument("--embed-dim", type=int, default=64, help="嵌入维度")
    parser.add_argument("--n-users", type=int, default=50, help="合成用户数")
    parser.add_argument("--device", default="cuda", help="设备: cuda/cpu")
    parser.add_argument("--sigma", type=float, default=0.0, help="差分隐私噪声(0=关闭)")
    parser.add_argument("--early-stop", type=int, default=3, help="早停patience")
    parser.add_argument("--version", default="v0.1.0-synthetic", help="模型版本号")
    parser.add_argument(
        "--ckpt-dir",
        default=os.path.join(PROJECT_ROOT, "backend/app/services/ranking/checkpoints"),
        help="Checkpoint输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"POI深度推荐模型训练")
    print(f"版本: {args.version}")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # 检查设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA不可用，回退到CPU")
        device = "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # ---- Step 1: 生成合成数据 ----
    print("【Step 1】生成合成训练数据...")
    gen = SyntheticDataGenerator(args.data_root, seed=42)
    interactions, trajectories, meta = gen.prepare(
        n_users=args.n_users, n_traj_per_user=5
    )
    print(f"  POI数量: {meta['num_pois']}")
    print(f"  标签词表: {meta['num_tags']}个标签")
    print(f"  合成用户: {meta['num_users']}")
    print(f"  交互样本: {meta['num_interactions']}")
    print(f"  轨迹序列: {meta['num_trajectories']}")

    # 保存tag_vocab和poi_list供后续服务使用
    tag_vocab_path = os.path.join(args.ckpt_dir, "tag_vocab.json")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(tag_vocab_path, "w", encoding="utf-8") as f:
        json.dump({"tag_vocab": meta["tag_vocab"], "poi_list": meta["poi_list"]},
                  f, ensure_ascii=False, indent=2)
    print(f"  词表已保存: {tag_vocab_path}\n")

    # ---- Step 2: 切分训练/验证集 ----
    print("【Step 2】切分训练/验证集...")
    # 按用户分组: 80%用户训练, 20%验证
    all_users = list(range(meta["num_users"]))
    split = int(len(all_users) * 0.8)
    train_users = set(all_users[:split])
    val_users = set(all_users[split:])

    train_data = [i for i in interactions if i.user_id in train_users]
    val_data = [i for i in interactions if i.user_id in val_users]
    print(f"  训练集: {len(train_data)}条 ({len(train_users)}用户)")
    print(f"  验证集: {len(val_data)}条 ({len(val_users)}用户)\n")

    # ---- Step 3: 创建模型 ----
    print("【Step 3】创建深度推荐模型...")
    model = POIDeepRecommender(
        num_tags=meta["num_tags"],
        embed_dim=args.embed_dim,
        window_size=10,
        rho=0.5,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}\n")

    # ---- Step 4: 训练 ----
    print("【Step 4】开始训练...")
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tau": 0.07,
        "gamma": 0.5,
        "lambda_cl": 0.1,
        "eta": 1e-5,
        "sigma": args.sigma,
        "mu": 0.001,
        "early_stop_patience": args.early_stop,
        "version": args.version,
        "embed_dim": args.embed_dim,
    }

    trainer = Trainer(
        model=model, config=config,
        tag_vocab=meta["tag_vocab"],
        poi_list=meta["poi_list"],
        device=device,
    )

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        all_interactions=interactions,
        ckpt_dir=args.ckpt_dir,
    )

    # ---- Step 5: 保存训练日志 ----
    log_path = os.path.join(args.ckpt_dir, f"training_log_{args.version}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": args.version,
            "config": config,
            "device": device,
            "model_params": {"total": total_params, "trainable": trainable_params},
            "data_stats": {
                "num_pois": meta["num_pois"],
                "num_tags": meta["num_tags"],
                "num_users": meta["num_users"],
                "num_interactions": meta["num_interactions"],
                "num_trajectories": meta["num_trajectories"],
                "train_size": len(train_data),
                "val_size": len(val_data),
            },
            "training_history": history,
            "best_ndcg": trainer.best_ndcg,
            "trained_at": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n训练日志已保存: {log_path}")

    # ---- Step 6: 生成训练曲线图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_list = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss曲线
        axes[0].plot(epochs_list, [h["train_loss"] for h in history], "b-o", label="Train Loss")
        axes[0].plot(epochs_list, [h["val_loss"] for h in history], "r-s", label="Val Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # NDCG曲线
        axes[1].plot(epochs_list, [h["val_ndcg"] for h in history], "g-^", label="Val NDCG@10")
        axes[1].set_title("NDCG@10 on Validation Set")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("NDCG@10")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 损失组分分解
        axes[2].plot(epochs_list, [h["train_L_rec"] for h in history], "b-o", label="L_rec (BCE)")
        axes[2].plot(epochs_list, [h["train_L_cl"] for h in history], "r-s", label="L_cl (Contrastive)")
        axes[2].set_title("Loss Components Breakdown")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss Value")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        curve_path = os.path.join(args.ckpt_dir, f"training_curve_{args.version}.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"训练曲线图已保存: {curve_path}")
    except ImportError:
        print("[WARN] matplotlib未安装，跳过曲线图生成。安装: pip install matplotlib")

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  最佳NDCG@10: {trainer.best_ndcg:.4f}")
    print(f"  模型版本: {args.version}")
    print(f"  Checkpoint: {os.path.join(args.ckpt_dir, 'best.pt')}")
    print(f"  训练日志: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
