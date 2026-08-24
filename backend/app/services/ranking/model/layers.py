# 通用工具层：球面距离计算、安全归一化
import math
import torch


def haversine_distance(loc1, loc2):
    """
    球面距离(Haversine公式)，支持标量对和Tensor批处理
    loc1/loc2: (lat, lng) 元组 或 Tensor[2]
    返回: 距离(km)
    """
    if isinstance(loc1, (list, tuple)):
        lat1, lng1 = float(loc1[0]), float(loc1[1])
        lat2, lng2 = float(loc2[0]), float(loc2[1])
        R = 6371.0
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
        return 2 * R * math.asin(min(1.0, math.sqrt(a)))
    else:
        # Tensor标量版
        R = 6371.0
        lat1 = loc1[0] * math.pi / 180
        lat2 = loc2[0] * math.pi / 180
        dlat = (loc2[0] - loc1[0]) * math.pi / 180
        dlng = (loc2[1] - loc1[1]) * math.pi / 180
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
        return 2 * R * torch.asin(torch.sqrt(torch.clamp(a, min=1e-12)))


def haversine_distance_batch(loc1, loc2):
    """
    批处理球面距离
    loc1, loc2: [batch, 2] (lat, lng)
    返回: [batch] 距离(km)
    """
    R = 6371.0
    lat1 = loc1[:, 0] * math.pi / 180
    lat2 = loc2[:, 0] * math.pi / 180
    dlat = (loc2[:, 0] - loc1[:, 0]) * math.pi / 180
    dlng = (loc2[:, 1] - loc1[:, 1]) * math.pi / 180
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
    return 2 * R * torch.asin(torch.sqrt(torch.clamp(a, min=1e-12)))


def safe_normalize_tensor(values):
    """Tensor批处理Min-Max归一化到[0,1]，带边界保护"""
    if values.numel() == 0:
        return torch.full_like(values, 0.5)
    min_val = values.min()
    max_val = values.max()
    if (max_val - min_val).abs() < 1e-8:
        return torch.full_like(values, 0.5)
    return (values - min_val) / (max_val - min_val)


def safe_normalize_distance_tensor(distances_m):
    """距离反向归一化（越近得分越高），输入为米"""
    if distances_m.numel() == 0:
        return torch.full_like(distances_m, 0.5)
    min_val = distances_m.min()
    max_val = distances_m.max()
    if (max_val - min_val).abs() < 1e-8:
        return torch.full_like(distances_m, 0.5)
    return 1 - (distances_m - min_val) / (max_val - min_val)
