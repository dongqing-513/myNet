import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sacred import Experiment
import torch

ex = Experiment("TVLT")

def _loss_names(d):
    ret = {
        "vam": 0,
        "vatr": 0,
        "vtm": 0,
        "mae_audio": 0,
        "mae_video": 0,
        "vqa": 0,
        "mlm": 0,
        "mosei": 0,
        "moseiemo": 0,
    }
    ret.update(d)
    return ret

#Sacred要求所有配置项必须在基础配置函数（@ex.config）中定义默认值
@ex.config
def config():
    exp_name = "TVLT"
    seed = 0
    datasets = []
    loss_names = _loss_names({})
    # batch_size = 4096  # 将批量大小调整为64 this is a desired batch size; pl trainer will accumulate gradients when per step bat
    batch_size = 32

    max_text_len = 40
    draw_false_text = 0
    tokenizer = "bert-base-uncased" # tokenizer for text
    bert_model = "bert-base-uncased" # bert model path
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    use_text = False

    # Video setting
    video_size = 224 # video frame reshape size
    draw_false_video = 0 # draw negative video for video-audio matching
    video_only = False
    max_frames = 64 # max frames of frame position embedding
    num_frames = 8 # number frames to use for input video
    use_video = True

    # Audio Setting
    audio_size = 1024 # max audio spectrogram
    frequency_size = 128 # frequency axis size
    max_audio_patches = 1020 # max temporal position embedding
    draw_false_audio = 0 # draw negative audio
    use_audio = True
    audio_only = False
    frame_masking = False # frame level audio masking

    # Transformer Setting
    model_type = "mae_vit_base_patch16_dec512d8b" # model configuration
    patch_size = int(16)
    audio_patch_size = [int(16), int(16)]
    hidden_size = int(768)
    decoder_hidden_size = int(512)
    num_heads = int(12)  # 增加注意力头数，从8增加到12
    num_layers = int(12)
    mlp_ratio = float(4.0)
    use_mae = bool(False)
    drop_rate = float(0.1)  # 降低dropout率到0.1
    fusion_type = str('gate')  # 融合方式：'concat', 'add', 'gate'
    skip_interval = int(1)  # 跳跃连接间隔，每隔几层添加一次跳跃连接
    normalize_before = bool(True)  # 是否在attention和FFN之前进行归一化
    attn_mask = bool(False)  # 是否使用注意力掩码
    relu_dropout = float(0.15)  # ReLU层的dropout，降低到0.1
    res_dropout = float(0.15)  # 残差连接的dropout，降低到0.1

    # 注意力机制参数
    num_groups = int(4)  # 分组线性变换的分组数
    reduction_ratio = int(16)  # 特征重校准的压缩比

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 5e-5       # 降低基础学习率，避免训练不稳定
    weight_decay = 0.01        # 减小权重衰减
    decay_power = 1
    max_epoch = 15  # 增加训练轮数
    max_steps = 1000000
    warmup_steps = 3000
    warmup_ratio = 0.05
    beta1 = 0.9
    beta2 = 0.999             # 使用更保守的beta2值
    eps = 1e-8                # 提高数值稳定性

    # 学习率调度器设置
    lr_scheduler = "cosine_warmup"
    min_lr_ratio = 0.01       # 提高最小学习率比例，避免后期学习过慢

    # Dropout和正则化设置
    attention_dropout = 0.1    # 降低dropout率
    hidden_dropout = 0.1      # 降低dropout率
    drop_rate = 0.1           # 统一降低dropout率

    # 梯度裁剪
    gradient_clip_val = 0.5     # 使用更保守的梯度裁剪
    gradient_clip_val_msaf = 0.3  # 对MSAF使用更小的裁剪阈值

    # 早停设置
    early_stopping_patience = 5    # 减少耐心值，及时停止过拟合
    early_stopping_min_delta = 0.001  # 添加最小改善阈值

    # 验证设置
    val_check_interval = 0.5  # 增加验证频率

    # 批次设置
    accumulate_grad_batches = 32   # 增加梯度累积来模拟更大批次

    # Checkpoint settings
    ckpt_path = None  # Path to resume training from
    save_top_k = 3    # Number of best checkpoints to keep
    monitor = "val_loss"  # Metric to monitor
    save_last = True  # Save the last checkpoint
    every_n_epochs = 1  # Save checkpoint frequency
    ckpt_dir = "/data1/checkpoints/MyNet"  # Directory to save checkpoints

    # Training robustness
    max_time = "12:00:00"  # Maximum training time
    auto_resume = True     # Automatically resume from checkpoint

    # Downstream Setting
    vqav2_label_size = 3129
    get_va_recall_metric = False # perform audio video retrieval at end of each epoch

    # PL Trainer Setting
    fast_dev_run = False
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 1  # you should define this manually with per_gpu_batch_size=#
    gpus = 2
    # gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_nodes = 1
    strict_load = False
    load_local_path = ""
    load_hub_path = ""
    num_workers = 16


@ex.named_config
def task_cls_moseiemo():
    exp_name = "cls_moseiemo"
    datasets = ["moseiemo"]
    loss_names = _loss_names({"moseiemo": 1})
    model_type='mae_vit_base_patch16_dec512d8b'
    batch_size = 128
    audio_size = 1024
    num_frames = 8
    use_video = True
    use_audio = True
    use_text = False
    learning_rate = 1e-4
    max_epoch = 10

# 在具体的任务配置（如task_cls_mosei）中可以覆盖这个默认值
@ex.named_config
def task_cls_mosei():
    exp_name = "cls_mosei"
    datasets = ["mosei"]
    loss_names = _loss_names({"mosei": 1})
    model_type= 'mae_vit_base_patch16_dec512d8b' # for32'mae_vit_base_patch16_dec512d8b'

    # 当前值：128
    # 建议调整为64，减小批量大小可以提高模型泛化能力，避免后期过拟合
    batch_size = 64

    audio_size = 1024

    num_frames = 8

    use_video = True
    use_audio = True
    use_text = True

    learning_rate = 3e-5       # 使用更保守的学习率
    max_epoch = 15             # 增加训练轮数

    tokenizer = "/home/mz/demo/MyNet/mybert/models--bert-base-uncased"
    bert_model = "/home/mz/demo/MyNet/bert"

    # 词汇表的大小。可以识别的不同单词或标记的总数。
    vocab_size = 30522
    # 输入文本的最大长度。如果文本长度超过这个值，可能会被截断。表示最长的文本长度为 768 个标记或字符。
    max_text_len = 197 # 512 (audio patch size (2,128))225 78 220
    # 是否进行全词掩码。以整个单词为单位进行掩码或单个标记
    whole_word_masking = False
    # 随机掩码标记的概率。有 0.15 的概率会随机选择一个标记进行掩码，然后让模型预测被掩码的标记
    mlm_prob = 0.15
    # GPU的数量
    gpus = 2
    # 进行分布式训练
    num_nodes = 1


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    model_type='mae_vit_base_patch16_dec512d8b'
    batch_size = 128
    audio_size = 1024
    num_frames = 1
    use_video = True
    use_audio = True
    use_text = False
    learning_rate = 1e-5
    max_epoch = 10


@ex.named_config
def task_finetune_msrvtt():
    exp_name = "finetune_msrvtt"
    datasets = ["msrvtt"]
    loss_names = _loss_names({"vam": 1, "vatr": 1})
    batch_size = 128
    audio_size = 1024
    num_frames = 8
    use_video = True
    use_audio = True
    use_text = False
    get_va_recall_metric = True
    draw_false_video = 23
    learning_rate = 1e-5
    max_epoch = 40
    max_steps = 100000


@ex.named_config
def task_mae_vam():
    exp_name = "mae_vam"
    datasets = ["howto100m", "yttemporal"]
    loss_names = _loss_names({"vam": 1, "mae_video": 1, "mae_audio": 1})
    batch_size = 4096
    audio_size = 1024
    num_frames = 4
    use_video = True
    use_audio = True
    use_text = False
    draw_false_video = 1
    use_mae = True
    learning_rate = 1e-5
    max_epoch = 100
