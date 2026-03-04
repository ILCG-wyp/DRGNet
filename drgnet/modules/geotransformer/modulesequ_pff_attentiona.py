# modules/equivariant_ppf_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EquivariantPPFAttention(nn.Module):
    """
    旋转等变的PPF注意力模块
    关键：保持VN特征的旋转等变性，用PPF作为调制信号
    """

    def __init__(self, in_dim, out_dim, num_heads=4, hidden_dim=64):
        super(EquivariantPPFAttention, self).__init__()

        # 1. PPF编码器（旋转不变）
        self.ppf_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim // 3)  # 输出用于调制VN特征
        )

        # 2. 等变线性层（用于特征变换）
        from pareconv.modules.layers import VNLinear

        self.vn_query = VNLinear(in_dim, out_dim)
        self.vn_key = VNLinear(in_dim, out_dim)
        self.vn_value = VNLinear(in_dim, out_dim)

        # 3. 门控机制：用PPF信息控制特征融合
        self.gate_proj = nn.Linear(out_dim // 3, out_dim)

        self.num_heads = num_heads
        self.out_dim = out_dim

    def safe_atan2(self, y, x):
        """数值稳定的atan2计算"""
        # 确保分母不为零
        eps = 1e-8
        x_safe = torch.where(torch.abs(x) < eps, torch.sign(x) * eps, x)
        return torch.atan2(y, x_safe)

    def calc_ppf(self, points, point_normals, neighbor_points, neighbor_normals):
        """计算PPF特征（旋转不变）"""
        # points: [N, 3]
        # point_normals: [N, 3]
        # neighbor_points: [N, K, 3]
        # neighbor_normals: [N, K, 3]

        if points.dim() == 2:
            points = points.unsqueeze(1)  # [N, 1, 3]
        if point_normals.dim() == 2:
            point_normals = point_normals.unsqueeze(1)  # [N, 1, 3]

        points_expanded = points.expand(-1, neighbor_points.shape[1], -1)
        point_normals_expanded = point_normals.expand(-1, neighbor_points.shape[1], -1)

        vec_d = neighbor_points - points_expanded

        # 使用torch.norm（数值稳定）
        d = torch.norm(vec_d, dim=-1, keepdim=True)  # [N, K, 1]

        # angle(n1, vec_d)
        y = torch.sum(point_normals_expanded * vec_d, dim=-1, keepdim=True)  # [N, K, 1]
        x = torch.cross(point_normals_expanded, vec_d, dim=-1)  # [N, K, 3]
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # [N, K, 1]
        angle1 = self.safe_atan2(x_norm, y) / np.pi  # [N, K, 1]

        # angle(n2, vec_d)
        y = torch.sum(neighbor_normals * vec_d, dim=-1, keepdim=True)  # [N, K, 1]
        x = torch.cross(neighbor_normals, vec_d, dim=-1)  # [N, K, 3]
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # [N, K, 1]
        angle2 = self.safe_atan2(x_norm, y) / np.pi  # [N, K, 1]

        # angle(n1, n2)
        y = torch.sum(point_normals_expanded * neighbor_normals, dim=-1, keepdim=True)  # [N, K, 1]
        x = torch.cross(point_normals_expanded, neighbor_normals, dim=-1)  # [N, K, 3]
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # [N, K, 1]
        angle3 = self.safe_atan2(x_norm, y) / np.pi  # [N, K, 1]

        ppf = torch.cat([d, angle1, angle2, angle3], dim=-1)  # [N, K, 4]
        return ppf

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices, normals):
        """
        参数:
            q_pts: [N, 3] 查询点
            s_pts: [M, 3] 支持点
            s_feats: [M, D, 3] VN特征（旋转等变）
            neighbor_indices: [N, K] K近邻索引
            normals: [M, 3] 法向量
        返回:
            enhanced_feats: [N, out_dim, 3] 增强的VN特征
        """
        N, K = neighbor_indices.shape
        M = s_pts.shape[0]

        # 1. 计算PPF特征（旋转不变）
        neighbor_pts = s_pts[neighbor_indices]  # [N, K, 3]
        neighbor_norms = normals[neighbor_indices]  # [N, K, 3]

        # 获取查询点的法向量（使用第一个邻居的法向量作为近似）
        q_normals = normals[neighbor_indices[:, 0]]  # [N, 3]

        ppf = self.calc_ppf(q_pts, q_normals, neighbor_pts, neighbor_norms)  # [N, K, 4]

        # 2. 编码PPF特征
        ppf_encoded = self.ppf_encoder(ppf)  # [N, K, out_dim//3]

        # 3. 计算注意力权重（使用PPF信息）
        # 将PPF编码平均池化得到每个查询点的调制信号
        ppf_modulation = ppf_encoded.mean(dim=1)  # [N, out_dim//3]

        # 4. 计算门控权重
        gate_weights = torch.sigmoid(self.gate_proj(ppf_modulation))  # [N, out_dim]

        # 5. 获取邻居特征
        neighbor_feats = s_feats[neighbor_indices]  # [N, K, D, 3]

        # 6. 计算注意力（简化版：平均池化+门控）
        # 注意：这里我们保持VN特征的旋转等变性
        aggregated = neighbor_feats.mean(dim=1)  # [N, D, 3] - 保持等变性

        # 7. 通过VN线性层变换
        transformed = self.vn_value(aggregated)  # [N, out_dim, 3]

        # 8. 用PPF门控权重调制输出（按通道调制）
        # 将标量门控权重扩展到3D
        gate_weights_3d = gate_weights.unsqueeze(-1)  # [N, out_dim, 1]

        # 调制：保持每个特征向量的方向，只调整幅度
        modulated = transformed * gate_weights_3d  # [N, out_dim, 3]

        return modulated