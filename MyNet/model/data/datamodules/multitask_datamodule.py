import functools

from pytorch_lightning import LightningDataModule
# from pytorch_lightning.trainer.supporters import CombinedLoader

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules
          
class MTDataModule(LightningDataModule):
    # 配置字典、是否分布式训练
    def __init__(self, _config, dist=False):
        # MyNet/model/data/datamodules/__init__.py !!
        datamodule_keys = _config["datasets"] # 数据集列表
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()]

        self.batch_size = self.dms[0].batch_size
        self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist

    # 下载、预处理等一次性的数据准备操作
    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    # 遍历所有的数据集对象，调用它们的 setup 方法
    def setup(self, stage):
        
        for dm in self.dms:
            dm.setup(stage)
            
        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        self.tokenizer = self.dms[0].tokenizer

        self.collate = functools.partial(
            self.dms[0].train_dataset.collate, mlm_collator=self.dms[0].mlm_collator,
        )

        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
                
    def train_dataloader(self):
        # 数据集、批次大小、采样器、工作线程数和 collate_fn 作为参数，并返回相应的 DataLoader 对象
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def test_dataloader(self):
        
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader
