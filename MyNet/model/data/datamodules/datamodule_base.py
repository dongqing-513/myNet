import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)

# 获取预训练的 BertTokenizer
def get_pretrained_tokenizer(from_pretrained):
    # 是否已经初始化了分布式环境
    if torch.distributed.is_initialized():
        # 在分布式训练环境中，只有 rank 为 0 的进程会首先从指定的预训练路径加载 BertTokenizer
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        # 其他进程等待直到所有进程都到达同步点
        torch.distributed.barrier()
    # 最后所有进程都从相同的预训练路径加载 BertTokenizer。
    # 如果from_pretrained这个字符串中包含"uncased"，则表示应该进行小写转换。
    # 因为一些预训练模型是在小写文本上进行训练的，所以在使用这些模型时需要进行相应的文本预处理。
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class BaseDataModule(LightningDataModule):
    # MyNet/model/config.py ！
    def __init__(self, _config):
        super().__init__()
        # 从配置_config中提取各种参数
        self.data_dir = _config["data_root"]    # 数据目录

        self.num_workers = _config["num_workers"]   # 工作线程数
        self.batch_size = _config["per_gpu_batchsize"]  # 批次大小
        self.eval_batch_size = self.batch_size
        
        self.video_size = _config["video_size"] # 视频尺寸
        self.audio_size = _config["audio_size"] # 音频尺寸
        self.max_text_len = _config["max_text_len"] # 最大文本长度
        self.num_frames = _config["num_frames"]
        self.draw_false_audio = _config["draw_false_audio"]
        self.draw_false_video = _config["draw_false_video"]
        self.draw_false_text = _config["draw_false_text"]
        self.audio_only = _config["audio_only"]
        self.video_only = _config["video_only"]
        self.use_audio = _config["use_audio"]
        self.use_text = _config["use_text"]

        # 获取预训练的分词器，并设置为实例属性
        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        self.bert_model = _config["bert_model"]

        # 选择合适的数据整理器
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            # 以整个单词为单位进行掩码操作，确保被掩码的部分是完整的单词而不是单个字符或标记。
            else DataCollatorForLanguageModeling
            # 随机选择单个标记进行掩码。
        )

        # 创建对数据进行整理和预处理的对象
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            split="train",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=self.draw_false_audio,
            draw_false_video=self.draw_false_video,
            draw_false_text=self.draw_false_text,
            audio_only=self.audio_only,
            video_only=self.video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            split="val",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=self.draw_false_audio,
            draw_false_video=self.draw_false_video,
            draw_false_text=self.draw_false_text,
            audio_only=self.audio_only,
            video_only=self.video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                split="val",
                video_size=self.video_size,
                audio_size=self.audio_size,
                max_text_len=self.max_text_len,
                num_frames=self.num_frames,
                draw_false_audio=0,
                draw_false_video=0,
                draw_false_text=0,
                audio_only=self.audio_only,
                video_only=self.video_only,
                use_audio=self.use_audio,
                use_text=self.use_text,
            )

    def make_no_false_val_dset(self, image_only=False, video_only=False, audio_only=False):
        return self.dataset_cls(
            self.data_dir,
            split="val",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=0,
            draw_false_video=0,
            draw_false_text=0,
            audio_only=audio_only,
            video_only=video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            split="test",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=self.draw_false_audio,
            draw_false_video=self.draw_false_video,
            draw_false_text=self.draw_false_text,
            audio_only=self.audio_only,
            video_only=self.video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
