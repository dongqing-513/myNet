from model.data.datasets import MOSEIDataset
from .datamodule_base import BaseDataModule
from torch.utils.data import DataLoader


class MOSEIDataModule(BaseDataModule):
    # 继承BaseDataModule
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MOSEIDataset

    @property
    def dataset_cls_no_false(self):
        return MOSEIDataset

    @property
    def dataset_name(self):
        return "mosei"

    def train_dataloader(self):
        dataset = self.train_dataset
        sampler = dataset.get_sampler(self.batch_size) if hasattr(dataset, 'get_sampler') else None
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size if sampler is None else 1,  # 如果使用采样器，批次大小由采样器控制
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None,  # 如果使用采样器，不需要shuffle
            collate_fn=self.train_dataset.collate
        )
