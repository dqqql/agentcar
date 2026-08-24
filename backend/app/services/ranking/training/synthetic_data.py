# 合成数据生成器：从POI原始数据构造训练集（蒸馏自旧算法思想）
import json
import os
import random
import math
import glob
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class PoiRecord:
    poi_id: str
    name: str
    poi_type: str  # spot / food / hotel
    lat: float
    lng: float
    rating: float
    distance_m: float
    popularity: float
    tags: list[str] = field(default_factory=list)


@dataclass
class Interaction:
    user_id: int
    poi_idx: int
    s_obj: float
    label: float  # 1.0=强正 0.5=弱正 0.0=负
    user_tag_indices: list[int]
    poi_tag_indices: list[int]
    poi_lat: float
    poi_lng: float
    traj_id: int | None = None
    strong_pos_idx: int | None = None
    weak_pos_idx: int | None = None


@dataclass
class Trajectory:
    traj_id: int
    user_id: int
    poi_indices: list[int]
    poi_latlngs: list[tuple[float, float]]


class SyntheticDataGenerator:
    """从POI原始数据构造合成训练集（旧算法蒸馏）"""

    def __init__(self, data_root: str, seed: int = 42):
        self.data_root = data_root
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.pois: list[PoiRecord] = []
        self.tag_vocab: dict[str, int] = {}  # tag → index (1-based, 0=padding)
        self.users: list[dict] = []

    # ---- Step 1: 加载POI数据 ----

    def load_pois(self):
        """扫描所有detail.json，解析三种类型的POI"""
        patterns = [
            ("place", "scripts/getdata/place/output/*/detail.json"),
            ("food", "scripts/getdata/food/output/*/detail.json"),
            ("hotel", "scripts/getdata/hotel/output/*/detail.json"),
        ]
        for poi_type, pattern in patterns:
            search_path = os.path.join(self.data_root, pattern)
            for fpath in glob.glob(search_path):
                self._parse_detail_file(fpath, poi_type)
        return self.pois

    def _parse_detail_file(self, fpath: str, poi_type: str):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data.get("records", []):
            poi = self._normalize_record(rec, poi_type)
            if poi:
                self.pois.append(poi)

    def _normalize_record(self, rec: dict, poi_type: str) -> PoiRecord | None:
        tags: list[str] = []
        if poi_type == "place":
            poi_id = rec.get("id", "")
            lat = float(rec.get("lat") or 0)
            lng = float(rec.get("lon") or 0)
            rating = 4.0  # 景点默认评分
            raw_dist = rec.get("distance", 1000)
            if isinstance(raw_dist, list):
                distance_m = float(raw_dist[0]) if raw_dist else 1000.0
            else:
                distance_m = float(raw_dist) if raw_dist else 1000.0
            type_str = rec.get("type", "")
            tags = [t.strip() for t in type_str.split(";") if t.strip()]
            pop = 0.5
        elif poi_type == "food":
            poi_id = rec.get("poi_id", "")
            lat = float(rec.get("latitude") or 0)
            lng = float(rec.get("longitude") or 0)
            rating = float(rec.get("rating") or 4.0)
            distance_m = float(rec.get("distance_m") or 1000)
            cat = rec.get("category", "")
            tags = [t.strip() for t in cat.split(";") if t.strip()]
            raw_tags = rec.get("tags", "")
            if raw_tags:
                for part in raw_tags.split("|"):
                    for t in part.split(","):
                        t = t.strip()
                        if t:
                            tags.append(t)
            pop = float(rec.get("photo_count") or 0) / 10.0
        elif poi_type == "hotel":
            poi_id = rec.get("hotel_id", "")
            lat = float(rec.get("latitude") or 0)
            lng = float(rec.get("longitude") or 0)
            rating = float(rec.get("rating") or 4.0)
            distance_m = float(rec.get("distance_m") or 1000)
            amenities = rec.get("amenities", [])
            tag_map = {
                "wifi": "网络", "breakfast_option": "早餐", "pool": "游泳池",
                "gym": "健身", "parking": "停车", "family_friendly": "亲子",
                "business_area": "商务", "air_conditioning": "空调",
                "hot_water": "热水", "laundry": "洗衣", "metro_access": "地铁",
                "front_desk_24h": "前台24小时",
            }
            tags = [tag_map.get(a, a) for a in amenities]
            pop = float(rec.get("review_count", 0)) / 1000.0
        else:
            return None

        if not poi_id or lat == 0 or lng == 0:
            return None

        # 限制标签数量
        tags = tags[:10] if len(tags) > 10 else tags
        return PoiRecord(
            poi_id=poi_id, name=rec.get("name", ""), poi_type=poi_type,
            lat=lat, lng=lng, rating=rating, distance_m=distance_m,
            popularity=min(pop, 10.0), tags=tags,
        )

    # ---- Step 2: 构建标签词表 ----

    def build_tag_vocab(self):
        self.tag_vocab = {"<pad>": 0}
        for poi in self.pois:
            for tag in poi.tags:
                if tag not in self.tag_vocab:
                    self.tag_vocab[tag] = len(self.tag_vocab)
        return self.tag_vocab

    def tags_to_indices(self, tags: list[str]) -> list[int]:
        return [self.tag_vocab[t] for t in tags if t in self.tag_vocab] or [0]

    # ---- Step 3: 生成合成用户 ----

    def generate_users(self, n_users: int = 50):
        all_tags = list(self.tag_vocab.keys())
        all_tags.remove("<pad>")
        for uid in range(n_users):
            n_prefs = self.rng.randint(3, 6)
            prefs = self.rng.sample(all_tags, min(n_prefs, len(all_tags)))
            # 随机城市中心
            if self.pois:
                center_poi = self.rng.choice(self.pois)
                center = (center_poi.lat, center_poi.lng)
            else:
                center = (30.0, 120.0)
            self.users.append({
                "user_id": uid,
                "preferences": prefs,
                "center_lat": center[0],
                "center_lng": center[1],
            })
        return self.users

    # ---- Step 4: 生成交互数据（旧算法蒸馏） ----

    def generate_interactions(self) -> list[Interaction]:
        interactions: list[Interaction] = []
        for user in self.users:
            uid = user["user_id"]
            user_tag_idx = self.tags_to_indices(user["preferences"])

            # 为该用户计算所有POI的s_obj和s_sub
            poi_scores = []
            for poi_idx, poi in enumerate(self.pois):
                s_obj = self._compute_s_obj(uid, poi_idx)
                s_sub = self._compute_s_sub(user["preferences"], poi.tags)
                final = 0.6 * s_obj + 0.4 * s_sub
                poi_scores.append((poi_idx, final, s_obj))

            # 按分数排序，分配标签
            poi_scores.sort(key=lambda x: x[1], reverse=True)
            n = len(poi_scores)
            for rank, (poi_idx, score, s_obj) in enumerate(poi_scores):
                if rank < int(n * 0.2):
                    label = 1.0
                elif rank < int(n * 0.5):
                    label = 0.5
                else:
                    label = 0.0
                poi = self.pois[poi_idx]
                interactions.append(Interaction(
                    user_id=uid, poi_idx=poi_idx, s_obj=s_obj, label=label,
                    user_tag_indices=user_tag_idx,
                    poi_tag_indices=self.tags_to_indices(poi.tags),
                    poi_lat=poi.lat, poi_lng=poi.lng,
                ))
        return interactions

    def _compute_s_obj(self, user_id: int, poi_idx: int) -> float:
        """简化版客观得分（与service.py stage1一致）"""
        poi = self.pois[poi_idx]
        user = self.users[user_id]
        # 距离 = 用户中心到POI的球面距离
        dist_km = self._haversine(user["center_lat"], user["center_lng"], poi.lat, poi.lng)
        dist_m = dist_km * 1000

        # 简化归一化（全局统计）
        all_ratings = [p.rating for p in self.pois]
        all_dists = [self._haversine(u["center_lat"], u["center_lng"], p.lat, p.lng) * 1000
                     for u in self.users[:1] for p in self.pois]
        all_pops = [p.popularity for p in self.pois]

        norm_r = (poi.rating - min(all_ratings)) / (max(all_ratings) - min(all_ratings) + 1e-8)
        norm_d = 1 - (dist_m - min(all_dists)) / (max(all_dists) - min(all_dists) + 1e-8)
        norm_h = (poi.popularity - min(all_pops)) / (max(all_pops) - min(all_pops) + 1e-8)
        return 0.4 * norm_r + 0.3 * norm_d + 0.3 * norm_h

    def _compute_s_sub(self, user_prefs: list[str], poi_tags: list[str]) -> float:
        """简化版主观匹配（标签Jaccard相似度）"""
        if not user_prefs or not poi_tags:
            return 0.5
        inter = len(set(user_prefs) & set(poi_tags))
        union = len(set(user_prefs) | set(poi_tags))
        return inter / union if union > 0 else 0.5

    @staticmethod
    def _haversine(lat1, lng1, lat2, lng2):
        R = 6371.0
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
        return 2 * R * math.asin(min(1.0, math.sqrt(a)))

    # ---- Step 5: 生成轨迹序列 ----

    def generate_trajectories(self, n_per_user: int = 5) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        traj_id = 0
        for user in self.users:
            for _ in range(n_per_user):
                seq_len = self.rng.randint(5, 12)
                # 从用户中心附近开始，贪心选最近POI
                start_poi = self._nearest_poi(user["center_lat"], user["center_lng"], set())
                if start_poi is None:
                    continue
                seq_indices = [start_poi]
                visited = {start_poi}
                for _ in range(seq_len - 1):
                    last = self.pois[seq_indices[-1]]
                    nxt = self._nearest_poi(last.lat, last.lng, visited)
                    if nxt is None:
                        break
                    seq_indices.append(nxt)
                    visited.add(nxt)

                if len(seq_indices) >= 3:
                    poi_latlngs = [(self.pois[i].lat, self.pois[i].lng) for i in seq_indices]
                    trajectories.append(Trajectory(
                        traj_id=traj_id, user_id=user["user_id"],
                        poi_indices=seq_indices, poi_latlngs=poi_latlngs,
                    ))
                    traj_id += 1
        return trajectories

    def _nearest_poi(self, lat: float, lng: float, visited: set[int]) -> int | None:
        best_idx, best_dist = None, float("inf")
        for idx, poi in enumerate(self.pois):
            if idx in visited:
                continue
            d = self._haversine(lat, lng, poi.lat, poi.lng)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    # ---- Step 6: 绑定对比学习样本对 ----

    def assign_contrastive_pairs(self, interactions: list[Interaction],
                                 trajectories: list[Trajectory]):
        """为每个交互绑定强正样本（同轨迹下一POI）和弱正样本（同用户不同POI）"""
        # 建立轨迹索引: (user_id, poi_idx) → traj_id → next_poi_interaction_idx
        traj_map: dict[int, dict[int, int]] = {}  # traj_id → {poi_idx → position in seq}
        for traj in trajectories:
            traj_map[traj.traj_id] = {poi_idx: pos for pos, poi_idx in enumerate(traj.poi_indices)}

        # 为每个interaction找traj_id
        inter_lookup = {}  # (user_id, poi_idx) → interaction index
        for i, inter in enumerate(interactions):
            inter_lookup[(inter.user_id, inter.poi_idx)] = i

        # 分配traj_id和strong_pos
        for traj in trajectories:
            for pos, poi_idx in enumerate(traj.poi_indices):
                key = (traj.user_id, poi_idx)
                if key in inter_lookup:
                    inter_i = inter_lookup[key]
                    interactions[inter_i].traj_id = traj.traj_id
                    # 强正样本: 序列中的下一个POI
                    if pos + 1 < len(traj.poi_indices):
                        next_key = (traj.user_id, traj.poi_indices[pos + 1])
                        if next_key in inter_lookup:
                            interactions[inter_i].strong_pos_idx = inter_lookup[next_key]

        # 分配弱正样本: 同用户的随机其他正样本
        user_positives: dict[int, list[int]] = {}
        for i, inter in enumerate(interactions):
            if inter.label > 0:
                user_positives.setdefault(inter.user_id, []).append(i)

        for inter_i, inter in enumerate(interactions):
            if inter.label > 0 and inter.user_id in user_positives:
                candidates = [j for j in user_positives[inter.user_id]
                              if j != inter_i and j != inter.strong_pos_idx]
                if candidates:
                    inter.weak_pos_idx = self.rng.choice(candidates)

    # ---- 主入口 ----

    def prepare(self, n_users: int = 50, n_traj_per_user: int = 5
                ) -> tuple[list[Interaction], list[Trajectory], dict]:
        self.load_pois()
        self.build_tag_vocab()
        self.generate_users(n_users)
        interactions = self.generate_interactions()
        trajectories = self.generate_trajectories(n_traj_per_user)
        self.assign_contrastive_pairs(interactions, trajectories)

        meta = {
            "num_pois": len(self.pois),
            "num_tags": len(self.tag_vocab),
            "num_users": len(self.users),
            "num_interactions": len(interactions),
            "num_trajectories": len(trajectories),
            "tag_vocab": self.tag_vocab,
            "poi_list": [{"poi_id": p.poi_id, "name": p.name, "lat": p.lat, "lng": p.lng,
                          "tags": p.tags, "rating": p.rating, "poi_type": p.poi_type,
                          "distance_m": p.distance_m, "popularity": p.popularity}
                         for p in self.pois],
        }
        return interactions, trajectories, meta
