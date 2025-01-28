import torch
# from pytorch_lightning.metrics import Metric
from torchmetrics import Metric

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # 用于记录预测正确的样本数量
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # 记录总的样本数量
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):        
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        else:
            raise ValueError("Expected logits to be a torch.Tensor")
        
        logits, target = (
            torch.tensor(logits).to(self.correct.device),
            torch.tensor(target).to(self.correct.device),
        )
        # 多分类
        if logits.size(1) > 1:
            # 每个样本在预测结果中概率最大的类别索引
            preds = logits.argmax(dim=-1).squeeze()
        # 二分类
        else:
            # 通过判断 logits 是否大于等于 0 来确定预测类别
            preds = (logits >= 0).squeeze()
        target = target.squeeze()    
        preds = preds.to(target.device)
        #  target 中值为 -100 的样本是无效样本，通过索引筛选，将预测值和真实标签中对应的无效样本都排除掉，只保留有效样本用于后续的准确率计算。
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape
        # 通过比较预测值 preds 和真实标签 target，找出两者相等的元素，
        # 然后使用 torch.sum 对这些相等元素的数量进行求和，并累加到 self.correct 变量中，从而更新预测正确的样本数量。
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        # 预测正确的样本数量 self.correct 除以总的有效样本数量 self.total，得到准确率指标的值
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class F1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # True Positives，正确预测为正类的样本数
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # False Positives，错误预测为正类的样本数
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # False Negatives，错误预测为负类的样本数
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            torch.tensor(logits.detach().cpu().numpy()).to(self.tp.device),
            torch.tensor(target.detach().cpu().numpy()).to(self.tp.device),
        )
        
        # 处理one-hot编码的输入
        if logits.dim() > 1 and logits.shape[-1] == 2:
            preds = logits.argmax(dim=-1).reshape(-1)  # 确保是1D张量
        else:
            preds = (logits >= 0).squeeze().reshape(-1)  # 确保是1D张量
            
        if target.dim() > 1 and target.shape[-1] == 2:
            target = target.argmax(dim=-1).reshape(-1)  # 确保是1D张量
        else:
            target = target.squeeze().reshape(-1)  # 确保是1D张量
        
        # 处理无效样本
        valid_mask = target != -100
        preds = preds[valid_mask]
        target = target[valid_mask]
        
        if target.numel() == 0:
            return 1
            
        assert preds.shape == target.shape, f"Shape mismatch: preds {preds.shape} != target {target.shape}"
        
        # 计算True Positives, False Positives, False Negatives
        self.tp += torch.sum((preds == 1) & (target == 1))
        self.fp += torch.sum((preds == 1) & (target == 0))
        self.fn += torch.sum((preds == 0) & (target == 1))

    def compute(self):
        # 计算精确率和召回率
        precision = self.tp / (self.tp + self.fp + 1e-7)  # 添加小值避免除零
        recall = self.tp / (self.tp + self.fn + 1e-7)
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
