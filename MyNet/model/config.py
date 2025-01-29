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


@ex.config
def config():
    exp_name = "TVLT"
    seed = 0
    datasets = []
    loss_names = _loss_names({})
    # batch_size = 4096  # 将批量大小调整为64 this is a desired batch size; pl trainer will accumulate gradients when per step bat
    batch_size = 16

    max_text_len = 40
    draw_false_text = 0
    tokenizer = "bert-base-uncased" # tokenizer for text
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
    num_heads = int(12)
    num_layers = int(12)
    mlp_ratio = float(4.0)
    use_mae = bool(False)
    drop_rate = float(0.2)  # 用于所有dropout参数 当前值0.1
    fusion_type = str('concat')  # 融合方式：'concat', 'add', 'gate'
    skip_interval = int(1)  # 跳跃连接间隔，每隔几层添加一次跳跃连接
    normalize_before = bool(True)  # 是否在attention和FFN之前进行归一化
    
    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4      # 提高基础学习率，让TVLT能更好学习
    weight_decay = 0.01       # 适当降低权重衰减
    decay_power = 1
    max_epoch = 20       # 增加训练轮数
    max_steps = 1000000
    warmup_steps = 2000      # 增加预热步数
    warmup_ratio = 0.05      # 降低预热比例，更平缓的开始
    beta1 = 0.9
    beta2 = 0.999            # 使用更标准的beta2值
    eps = 1e-8
    
    # 学习率调度器设置
    lr_scheduler = "cosine_warmup"
    min_lr_ratio = 0.001     # 降低最小学习率比例，避免后期学习停滞
    
    # Dropout和正则化设置
    attention_dropout = 0.1   # 降低dropout率，因为模型较大
    hidden_dropout = 0.1
    drop_rate = 0.1          # 统一降低dropout率
    
    # 模型结构设置
    fusion_type = 'concat'     # 使用门控机制进行特征融合
    skip_interval = 2        # 增加跳跃连接间隔
    normalize_before = True   # 保持在attention和FFN之前进行归一化
    
    # 梯度裁剪
    gradient_clip_val = 1.0   # 对TVLT使用较大的裁剪阈值
    gradient_clip_val_msaf = 0.5  # 对MSAF使用较小的裁剪阈值
    
    # 早停设置
    early_stopping_patience = 5    # 减少耐心值，及时停止过拟合
    early_stopping_min_delta = 0.001  # 添加最小改善阈值
    
    # 验证设置
    val_check_interval = 0.5  # 增加验证频率
    
    # 批次设置
    accumulate_grad_batches = 16  # 使用梯度累积来模拟大批次
    
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
    
    # 增加视频帧数以捕获更多时序信息，提高情感分析准确性
    num_frames = 8
    
    use_video = True
    use_audio = True 
    use_text = True
    
    # 当前值：1e-4
    # 建议调整为5e-5，降低学习率可以使模型更稳定地收敛，避免后期震荡
    learning_rate = 5e-5
    
    # 当前值：10
    # 建议调整为15，增加训练轮次配合较小的学习率，使模型更充分学习
    max_epoch = 10
    
    # Cross-modal skip connections configuration
    # use_skip_connections = True
    skip_interval = 2  # 跳跃连接间隔，每隔几层添加一次跳跃连接
    #skip_connection_type = "concat"  # Options: concat, add, gate
    
    tokenizer = "/home/mz/demo/MyNet/mybert/models--bert-base-uncased"
    # bert_model = "/home/mz/demo/MyNet/bert"
    # 词汇表的大小。可以识别的不同单词或标记的总数。表示模型的词汇表中有 768 个不同的元素。
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
    max_steps = 100000  # 改为整数形式
    
    
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
    
   
