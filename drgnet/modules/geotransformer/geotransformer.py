import pdb
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# === 新增：添加导入路径 ===
experiment_dir = os.path.join(os.path.dirname(__file__), '../../../experiment/3DMatch')
sys.path.insert(0, experiment_dir)

# 先导入必要的模块
from pareconv.modules.ops import pairwise_distance
from pareconv.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer

# === 首先定义原有的类 ===
class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)

        # ref_vectors = normals.unsqueeze(2)

        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)
        # a_embeddings = a_embeddings[:, :, :, 0, :]
        embeddings = d_embeddings + a_embeddings

        return embeddings

class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True, parallel=False
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list

# === 然后定义增强相关的类 ===
try:
    from positional_encoding import FusedGeometricStructureEmbedding
    print("✅ Successfully imported FusedGeometricStructureEmbedding from positional_encoding.py")
except ImportError:
    # 如果导入失败，在这里直接定义FusedGeometricStructureEmbedding
    print("Warning: Cannot import FusedGeometricStructureEmbedding from positional_encoding.py, defining it locally")

    class PPFStructualEmbedding(nn.Module):
        def __init__(self, hidden_dim, mode='local'):
            super(PPFStructualEmbedding, self).__init__()
            if mode == 'local':
                self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
                self.proj = nn.Linear(4, hidden_dim)
            elif mode == 'global':
                self.embedding = SinusoidalPositionalEmbedding(hidden_dim // 4)
                self.proj = nn.Linear(hidden_dim, hidden_dim)
            else:
                raise ValueError('mode should be in [local, global]')
            self.mode = mode

        def forward(self, ppf):
            if self.mode == 'local':
                embeddings = self.proj(ppf)
            elif self.mode == 'global':
                d_embeddings = self.embedding(ppf[..., 0])
                a_embeddings0 = self.embedding(ppf[..., 1])
                a_embeddings1 = self.embedding(ppf[..., 2])
                a_embeddings2 = self.embedding(ppf[..., 3])
                embeddings = torch.cat([d_embeddings, a_embeddings0, a_embeddings1, a_embeddings2], dim=-1)
                embeddings = self.proj(embeddings)
                embeddings = nn.functional.normalize(embeddings, dim=-1, p=2)
            else:
                raise ValueError('mode should be in [local, global]')
            return embeddings

    class FusedGeometricStructureEmbedding(nn.Module):
        def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max', ppf_hidden_dim=64):
            super(FusedGeometricStructureEmbedding, self).__init__()

            # 原有的距离+角度编码 - 现在GeometricStructureEmbedding已经定义
            self.original_embedding = GeometricStructureEmbedding(
                hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a
            )

            # RoITr风格的PPF旋转不变编码
            self.ppf_embedding = PPFStructualEmbedding(ppf_hidden_dim, mode='global')

            # 融合投影层
            self.fusion_proj = nn.Linear(hidden_dim + ppf_hidden_dim, hidden_dim)

            self.hidden_dim = hidden_dim
            self.ppf_hidden_dim = ppf_hidden_dim

        def calc_ppf_gpu(self, points, point_normals, patches, patch_normals):
            """PPF计算函数"""
            if points.dim() == 2:
                points = points.unsqueeze(1)
            if point_normals.dim() == 2:
                point_normals = point_normals.unsqueeze(1)

            points_expanded = points.expand(-1, patches.shape[1], -1)
            point_normals_expanded = point_normals.expand(-1, patches.shape[1], -1)

            vec_d = patches - points_expanded
            d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True))

            # angle(n1, vec_d)
            y = torch.sum(point_normals_expanded * vec_d, dim=-1, keepdim=True)
            x = torch.cross(point_normals_expanded, vec_d, dim=-1)
            x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
            angle1 = torch.atan2(x, y) / np.pi

            # angle(n2, vec_d)
            y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
            x = torch.cross(patch_normals, vec_d, dim=-1)
            x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
            angle2 = torch.atan2(x, y) / np.pi

            # angle(n1, n2)
            y = torch.sum(point_normals_expanded * patch_normals, dim=-1, keepdim=True)
            x = torch.cross(point_normals_expanded, patch_normals, dim=-1)
            x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
            angle3 = torch.atan2(x, y) / np.pi

            ppf = torch.cat([d, angle1, angle2, angle3], dim=-1)
            return ppf

        def forward(self, points, normals):
            batch_size, num_points, _ = points.shape

            # 1. 原有的距离+角度编码
            original_embeddings = self.original_embedding(points)

            # 2. RoITr风格的PPF旋转不变编码
            expanded_points = points.unsqueeze(1).expand(batch_size, num_points, num_points, 3)
            expanded_normals = normals.unsqueeze(1).expand(batch_size, num_points, num_points, 3)

            # 为每个点对计算PPF
            ppf_features = self.calc_ppf_gpu(
                points.unsqueeze(2),
                normals.unsqueeze(2),
                expanded_points,
                expanded_normals
            )

            # 通过PPF编码器
            ppf_embeddings = self.ppf_embedding(ppf_features)

            # 3. 拼接融合
            fused_embeddings = torch.cat([original_embeddings, ppf_embeddings], dim=-1)

            # 4. 投影回原有维度
            fused_embeddings = self.fusion_proj(fused_embeddings)

            return fused_embeddings

class EnhancedGeometricTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            num_heads,
            blocks,
            sigma_d,
            sigma_a,
            angle_k,
            dropout=None,
            activation_fn='ReLU',
            reduction_a='max',
            ppf_hidden_dim=64,
    ):
        super(EnhancedGeometricTransformer, self).__init__()

        # 使用融合的位置编码
        self.embedding = FusedGeometricStructureEmbedding(
            hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a,
            ppf_hidden_dim=ppf_hidden_dim
        )

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout,
            activation_fn=activation_fn, return_attention_scores=True
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        # 记录是否收到过警告
        self.warned_no_normals = False

    def forward(
            self,
            ref_points,
            src_points,
            ref_feats,
            src_feats,
            *args,  # 灵活接收参数
            **kwargs  # 灵活接收关键字参数
    ):
        """
        向后兼容的forward函数
        支持两种调用方式：
        1. 原始方式：transformer(points, points, feats, feats, masks, masks)
        2. 增强方式：transformer(points, points, feats, feats, normals, normals, masks, masks)
        """

        # 解析参数
        ref_masks = None
        src_masks = None
        ref_normals = None
        src_normals = None

        # 检查参数数量
        if len(args) >= 2:
            # 前两个额外参数可能是normals或masks
            arg1, arg2 = args[0], args[1]

            # 启发式判断：如果形状与points匹配且是浮点类型，可能是normals
            if (arg1.shape == ref_points.shape and arg1.dtype in [torch.float32, torch.float64] and
                    arg2.shape == src_points.shape and arg2.dtype in [torch.float32, torch.float64]):
                ref_normals, src_normals = arg1, arg2

                # 检查是否有额外的mask参数
                if len(args) >= 4:
                    ref_masks, src_masks = args[2], args[3]
            else:
                # 可能是masks
                ref_masks, src_masks = arg1, arg2

        # 检查关键字参数
        if 'ref_normals' in kwargs:
            ref_normals = kwargs['ref_normals']
        if 'src_normals' in kwargs:
            src_normals = kwargs['src_normals']
        if 'ref_masks' in kwargs:
            ref_masks = kwargs['ref_masks']
        if 'src_masks' in kwargs:
            src_masks = kwargs['src_masks']

        # 如果没有提供法向量，创建零向量（向后兼容）
        if ref_normals is None:
            ref_normals = torch.zeros_like(ref_points)
            if not self.warned_no_normals:
                print("[EnhancedGeometricTransformer] Warning: No ref_normals provided, using zeros")
                self.warned_no_normals = True

        if src_normals is None:
            src_normals = torch.zeros_like(src_points)
            if not self.warned_no_normals:
                print("[EnhancedGeometricTransformer] Warning: No src_normals provided, using zeros")

        # 使用融合编码（现在总是有normals，即使它们是零）
        ref_embeddings = self.embedding(ref_points, ref_normals)
        src_embeddings = self.embedding(src_points, src_normals)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list