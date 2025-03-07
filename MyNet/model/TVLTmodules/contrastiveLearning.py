import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearning(nn.Module):
    """
    通用对比学习模块，支持多模态对比学习
    """
    def __init__(
        self,
        input_dim,                # 输入特征维度
        projection_dim=128,       # 投影空间维度
        temperature=0.07,         # 温度参数
        modality_names=None,      # 模态名称列表
        projection_hidden_dim=None, # 投影头隐藏层维度，如果为None则使用input_dim
        use_hardneg=False,        # 是否使用硬负样本挖掘
        hardneg_samples=10,       # 硬负样本数量
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_hardneg = use_hardneg
        self.hardneg_samples = hardneg_samples

        # 默认的隐藏层维度与输入维度相同
        if projection_hidden_dim is None:
            projection_hidden_dim = input_dim

        # 如果未指定模态名称，使用默认名称
        if modality_names is None:
            modality_names = ["modality_1", "modality_2"]

        self.modality_names = modality_names

        # 为每个模态创建投影头
        self.projection_heads = nn.ModuleDict()
        for modality in modality_names:
            self.projection_heads[modality] = nn.Sequential(
                nn.Linear(input_dim, projection_hidden_dim),
                nn.ReLU(),
                nn.Linear(projection_hidden_dim, projection_dim)
            )

    def project_features(self, features_dict):
        """
        将不同模态的特征投影到对比学习空间

        Args:
            features_dict: 包含不同模态特征的字典，格式为 {模态名: 特征张量}
                           特征张量形状应为 [batch_size, input_dim]

        Returns:
            投影后的特征字典，格式为 {模态名: 投影特征张量}
            投影特征张量形状为 [batch_size, projection_dim]
        """
        projected_features = {}

        for modality, features in features_dict.items():
            if modality in self.projection_heads:
                # 投影特征
                projected = self.projection_heads[modality](features)
                # L2标准化
                projected = F.normalize(projected, p=2, dim=1)
                projected_features[modality] = projected

        return projected_features

    def compute_similarity_matrix(self, feat_a, feat_b):
        """
        计算两个特征集合之间的相似度矩阵

        Args:
            feat_a: 第一个特征集合，形状为 [batch_size, projection_dim]
            feat_b: 第二个特征集合，形状为 [batch_size, projection_dim]

        Returns:
            相似度矩阵，形状为 [batch_size, batch_size]
        """
        return torch.matmul(feat_a, feat_b.t()) / self.temperature

    def hard_negative_mining(self, sim_matrix):
        """
        执行硬负样本挖掘

        Args:
            sim_matrix: 相似度矩阵，形状为 [batch_size, batch_size]

        Returns:
            处理后的相似度矩阵
        """
        batch_size = sim_matrix.size(0)

        # 创建负样本掩码（对角线为0，其他位置为1）
        neg_mask = torch.eye(batch_size, device=sim_matrix.device) == 0

        # 只保留负样本部分
        sim_matrix_neg = sim_matrix * neg_mask

        # 对每行找出top-k个硬负样本
        hard_indices = torch.topk(sim_matrix_neg, k=min(self.hardneg_samples, batch_size-1), dim=1)[1]

        # 创建新的掩码，只保留正样本和硬负样本
        new_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=sim_matrix.device).unsqueeze(1).expand(-1, min(self.hardneg_samples, batch_size-1))
        new_mask[batch_indices.flatten(), hard_indices.flatten()] = True

        # 添加对角线（正样本）
        new_mask = new_mask | (torch.eye(batch_size, device=sim_matrix.device) == 1)

        # 应用掩码，将非硬负样本位置设为一个大的负值
        masked_sim_matrix = sim_matrix * new_mask.float() - 1e9 * (~new_mask)

        return masked_sim_matrix

    def compute_pairwise_loss(self, feat_a, feat_b, modality_a, modality_b):
        """
        计算一对模态之间的对比损失

        Args:
            feat_a: 第一个模态的特征，形状为 [batch_size, projection_dim]
            feat_b: 第二个模态的特征，形状为 [batch_size, projection_dim]
            modality_a: 第一个模态的名称
            modality_b: 第二个模态的名称

        Returns:
            对比损失值及详细信息
        """
        batch_size = feat_a.size(0)

        # 计算相似度矩阵
        sim_matrix = self.compute_similarity_matrix(feat_a, feat_b)

        # 如果启用硬负样本挖掘
        if self.use_hardneg and batch_size > 1:
            sim_matrix = self.hard_negative_mining(sim_matrix)

        # 创建标签 - 对角线元素为正样本对
        labels = torch.arange(batch_size, device=feat_a.device)

        # 双向对比损失
        loss_ab = F.cross_entropy(sim_matrix, labels)
        loss_ba = F.cross_entropy(sim_matrix.t(), labels)

        # 平均损失
        loss = (loss_ab + loss_ba) / 2

        return {
            'loss': loss,
            f'loss_{modality_a}_{modality_b}': loss_ab.item(),
            f'loss_{modality_b}_{modality_a}': loss_ba.item(),
            'sim_matrix': sim_matrix.detach()
        }

    def forward(self, features_dict):
        """
        计算所有模态对之间的对比损失

        Args:
            features_dict: 包含不同模态特征的字典，格式为 {模态名: 特征张量}
                           特征张量形状应为 [batch_size, input_dim]

        Returns:
            总体对比损失和详细信息
        """
        # 投影特征
        projected_features = self.project_features(features_dict)

        # 检查至少有两个模态
        if len(projected_features) < 2:
            return {'loss': torch.tensor(0.0, device=next(self.parameters()).device)}

        # 计算所有模态对之间的对比损失
        total_loss = 0.0
        loss_count = 0
        loss_details = {}

        # 获取所有模态对组合
        modality_pairs = []
        modalities = list(projected_features.keys())

        for i in range(len(modalities)):
            for j in range(i+1, len(modalities)):
                modality_pairs.append((modalities[i], modalities[j]))

        # 计算每对模态之间的损失
        for modality_a, modality_b in modality_pairs:
            feat_a = projected_features[modality_a]
            feat_b = projected_features[modality_b]

            # 跳过空特征或批次大小为0的情况
            if feat_a.size(0) == 0 or feat_b.size(0) == 0:
                continue

            # 计算这对模态之间的损失
            pair_results = self.compute_pairwise_loss(feat_a, feat_b, modality_a, modality_b)

            # 累加损失
            total_loss += pair_results['loss']
            loss_count += 1

            # 保存详细信息
            for k, v in pair_results.items():
                if k != 'loss':
                    loss_details[k] = v

        # 计算平均损失
        if loss_count > 0:
            total_loss = total_loss / loss_count

        # 返回结果
        results = {'loss': total_loss}
        results.update(loss_details)

        return results
