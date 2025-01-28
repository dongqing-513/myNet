import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
from .LowLevelAVFusionTransformer import BottleAttentionNet
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
from model.TVLTmodules import heads, objectives

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
"""
import sys
sys.path.append('/home/mz/demo/MyNet/NHFNet/networks')
from msaf_mosei import BottleAttentionNet
"""


class AudioPatchEmbed(nn.Module):
    """ Audio to Patch Embedding"""

    def __init__(
        self,
        img_size=173,
        patch_size=[16, 16],
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TVLT(nn.Module):
    # patch_size=16, audio_patch_size=[16, 16], embed_dim=768, depth=12, num_heads=12,
    # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    # mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    def __init__(
        self, img_size=224, in_chans=3,
        patch_size=16, audio_patch_size=[16, 16], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm), eps=1e-6,
        config=None,
    ):

        super().__init__()

        self.config = config

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        use_audio = config['use_audio']
        self.use_audio = use_audio
        self.use_mae = config["loss_names"]["mae_audio"] > 0 or config["loss_names"]["mae_video"] > 0
        self.patch_embed_v = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.num_patches_v = self.patch_embed_v.num_patches
        self.frequency_size = config["frequency_size"]
        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_size = patch_size
        self.temporal_embed = nn.Parameter(torch.zeros(
            1, config['max_frames'], config["hidden_size"]))
        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.num_patches_v, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_audio:
            self.patch_embed_a = AudioPatchEmbed(
                img_size=img_size,
                patch_size=audio_patch_size,#audio_patch_size=[2, 128]
                in_chans=1,
                embed_dim=embed_dim,
            )
            self.audio_patch_size = audio_patch_size
            self.type_embed_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_a = nn.Parameter(torch.zeros(
                1, config['max_audio_patches'], embed_dim))
            self.freq_patch_size = config['frequency_size']//audio_patch_size[1]
            self.freq_embed = nn.Parameter(torch.zeros(
                1, self.freq_patch_size, config["hidden_size"]))

        self.norm = norm_layer(embed_dim)

        if self.use_mae:
            self.decoder_pos_embed_v = nn.Parameter(
                torch.zeros(1, self.num_patches_v, decoder_embed_dim))
            self.decoder_temporal_embed = nn.Parameter(
                torch.zeros(1, config['max_frames'], decoder_embed_dim))
            self.decoder_embed = nn.Linear(
                embed_dim, decoder_embed_dim, bias=True)
            self.decoder_type_embed_v = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))
            self.mask_token_v = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_norm = norm_layer(decoder_embed_dim)
            if use_audio:
                self.decoder_type_embed_a = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim))
                self.decoder_pos_embed_a = nn.Parameter(torch.zeros(
                    1, config['max_audio_patches'], decoder_embed_dim))
                self.decoder_freq_embed = nn.Parameter(
                    torch.zeros(1, self.freq_patch_size, decoder_embed_dim))
                self.mask_token_a = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim))

        self.num_frames = config["num_frames"]
        self.max_audio_patches = config['max_audio_patches']
        self.frame_masking = config["frame_masking"]
        """
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        
        if self.use_mae:
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )
        """
        self.transformer = BottleAttentionNet()

        hs = config["hidden_size"]
        self.use_text = config['use_text']
        if config['use_text']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
            )
            self.text_embeddings = BertEmbeddings(bert_config)
            self.text_embeddings.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["vam"] > 0 or config["loss_names"]["vtm"] > 0:
            self.matching_score = heads.MatchingHead(config["hidden_size"])
            self.matching_score.apply(objectives.init_weights)
            
            if config["loss_names"]["vatr"] > 0:
                import copy
                self.rank_output = copy.deepcopy(self.matching_score)
                self.margin = 0.2
                for p in self.matching_score.parameters():
                    p.requires_grad = False

        if config["loss_names"]["mae_audio"] > 0:
            self.mae_score_audio = heads.MAEHead(
                config["decoder_hidden_size"], config['audio_patch_size'][0]*config['audio_patch_size'][1])
            self.audio_patch_size = config['audio_patch_size']
            self.mae_score_audio.apply(objectives.init_weights)

        if config["loss_names"]["mae_video"] > 0:
            self.patch_size = config['patch_size']
            self.num_patches = config['video_size']//config['patch_size']
            self.mae_score_video = heads.MAEHead(
                config["decoder_hidden_size"], config['patch_size']**2*3)
            self.mae_score_video.apply(objectives.init_weights)

        # ===================== Downstream ===================== #

        if config["loss_names"]["mosei"] > 0:
            vs = 1 # 单个输出维度 将模型的输出解释为一个连续的得分或概率
            self.classifier = nn.Sequential(
                # heads.Pooler(config["hidden_size"]), masf池化过了
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                # nn.Dropout(0.1), #防止过拟合 add
                nn.Linear(hs * 2, vs),
            )
            self.classifier.apply(objectives.init_weights)

        if config["loss_names"]["moseiemo"] > 0:
            self.classifier = nn.Sequential(
                heads.Pooler(config["hidden_size"]),
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 6),
            )
            self.classifier.apply(objectives.init_weights)

        if config["loss_names"]["vqa"] > 0:
            vs = config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                heads.Pooler(config["hidden_size"]),
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

    def init_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        std = 0.02

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.use_audio:
            w = self.patch_embed_a.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.cls_token, std=std)
        nn.init.normal_(self.temporal_embed, std=std)
        nn.init.normal_(self.type_embed_v, std=std)
        if self.use_audio:
            nn.init.normal_(self.freq_embed, std=std)
            nn.init.normal_(self.type_embed_a, std=std)

        if self.use_mae:
            nn.init.normal_(self.decoder_type_embed_v, std=std)
            nn.init.normal_(self.decoder_temporal_embed, std=std)
            nn.init.normal_(self.mask_token_v, std=std)

            if self.use_audio:
                nn.init.normal_(self.decoder_type_embed_a, std=std)
                nn.init.normal_(self.decoder_freq_embed, std=std)
                nn.init.normal_(self.mask_token_a, std=std)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed_v", "pos_embed_a", "cls_token", "mask_token_v", "mask_token_a", "temporal_embed", "decoder_pos_embed_v", "decoder_pos_embed_a"}

    def get_span_patch(audio_spans):
        patch_span = []
        patch_indexes = []
        for i in range(len(audio_spans)):
            span_i = []
            indexes_i = []
            for span in audio_spans[i]:
                s, t = torch.round(
                    span[0]/16).cpu().numpy(), torch.round(span[1]/16).cpu().numpy()
                span_i += [[s, t]]
                indexes_i += list(range(s, t))
            patch_span += [span_i]
            patch_indexes += [indexes_i]
        return patch_span, patch_indexes

    def random_masking_audio(self, x, att_mask=None, mask_ratio=0.15, audio_spans=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        F, T = 8, L//8  # frequency, time
        if audio_spans is not None:
            patch_span, patch_indexes = self.get_span_patch(audio_spans)
            len_keep = int(L * (1 - mask_ratio))
            noise = []
            for i in range(N):
                tmp_noise = torch.rand(len(patch_span[i]), device=x.device)
                noise_i = []
                for t in range(T):
                    if t in patch_indexes[i]:
                        noise_i += [tmp_noise[i, t]]
                    else:
                        noise_i += [torch.rand(1, device=x.device)[0]+1.0]
                noise += [noise_i]
            noise = torch.tensor(noise).to(x.device)
        else:
            len_keep = int(L * (1 - mask_ratio))
            # noise in [0, 1]
            noise = torch.rand(
                N, T, device=x.device).unsqueeze(-1).repeat(1, 1, F).view(N, L)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if att_mask is not None:
            mask *= att_mask

        att_mask = torch.gather(att_mask, dim=1, index=ids_keep)
        return x_masked, mask, ids_restore, att_mask

    def random_masking(self, x, att_mask=None, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if att_mask is not None:
            mask *= att_mask

        att_mask = torch.gather(att_mask, dim=1, index=ids_keep)
        return x_masked, mask, ids_restore, att_mask

    def cat_mask(self, mask_token, x, ids_restore):
        mask_tokens = mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        return x_

    def get_patch_mask(self, x):
        """
        masks out blank regions of the audios/images.
        """
        if len(x.shape) == 5:
            x = x.mean(2)
            x = F.avg_pool2d(x, self.patch_size,
                             self.patch_size).flatten(2).flatten(1)
            x_mask = x != -1
            return x_mask
        else:
            x = x.mean(1)
            x = F.avg_pool2d(x, self.audio_patch_size,
                             self.audio_patch_size).flatten(1)
            x_mask = x != -1
            return x_mask


    def forward(self, text_ids=None, text_masks=None, audio=None, audio_masks=None,
                video=None, video_masks=None, mask_visual=False, use_mae=False,
                audio_spans=None):
        # Dimension of input video tensor: (batch_size, number_of_frames, rgb_channel, width, height)  
        # 1, 8, 3, 224, 224
        # b, t, c, h, w = video.shape：在这里，b = 1（批次大小为 1），t = 8（8 个帧），c = 3（RGB 三个通道），h = 224（高度为 224），w = 224（宽度为 224
        # Dimension of input audio tensor: (batch_size, number_of_audio_channels, time, spectrogram)
        # B, C, H, W  torch.Size([1, 1, 464, 128])
        if text_ids is not None:
            text_embeds = self.text_embeddings(text_ids)
        else:
            text_embeds = None

            """
            x_a = self.patch_embed_a(audio)  #audio_patch_size=[2, 128]         
            x_a += self.freq_embed.repeat(1, x_a.size(1)//self.freq_patch_size, 1)
            x_a += torch.repeat_interleave(self.pos_embed_a[:, :x_a.size(
                1)//self.freq_patch_size], self.freq_patch_size, dim=1)
            x_a += self.type_embed_a
            """
        if audio is not None:
            # 获取音频输入的维度信息
            _, _, H, W = audio.shape
            x_a = self.patch_embed_a(audio)
            B, L, C = x_a.shape
            
            # 动态计算patch数量和大小
            freq_patches = H // self.audio_patch_size[0]
            time_patches = W // self.audio_patch_size[1]
            total_patches = freq_patches * time_patches
            
            # 计算频率嵌入的重复策略
            freq_embed_size = min(self.freq_embed.size(1), freq_patches)
            freq_repeat = (L + freq_embed_size - 1) // freq_embed_size
            
            # 截取并重复频率嵌入
            freq_embed = self.freq_embed[:, :freq_embed_size]
            freq_embed = freq_embed.repeat(1, freq_repeat, 1)[:, :L]
            
            # 位置嵌入也使用类似策略
            pos_embed_size = min(self.pos_embed_a.size(1), freq_repeat)
            pos_embed = self.pos_embed_a[:, :pos_embed_size]
            pos_embed = torch.repeat_interleave(pos_embed, freq_patches, dim=1)[:, :L]
            
            # 应用嵌入
            x_a = x_a + freq_embed + pos_embed + self.type_embed_a
            
            # 获取patch mask
            full_x_mask_a = self.get_patch_mask(audio)
                

        if video is not None:
            b, t, c, h, w = video.shape  # (128, 8, 3, 224, 224)
            # (128 * 8, 3, 224, 224)  = (1024, 3, 224, 224)
            # (224 / 16) * (224 / 16) = 14 * 14个图像块
            # (1024, 14 * 14, 768)    =  (1024, 196, 768)
            # parch embed
            x_v = self.patch_embed_v(video.reshape(b*t, c, h, w))
            # (128, 1568, 768)
            x_v = x_v.reshape(b, t * x_v.size(1), x_v.size(-1))       
            frame_patch_len = x_v.size(1)//t
            # (128, 1568, 768)
            # pos embed
            x_v += self.pos_embed_v.repeat(1, t, 1)
            x_v += torch.repeat_interleave(
                self.temporal_embed[:, :self.num_frames], frame_patch_len, dim=1)
            x_v += self.type_embed_v
            # (1, 1568, 768)
            full_x_mask_v = self.get_patch_mask(video)
        
        """
        if mask_visual:
            if video is not None:
                x_v, mask_v, ids_restore_v, enc_x_mask_v = self.random_masking(
                    x_v, full_x_mask_v)
            if audio is not None:
                if self.frame_masking:
                    x_a, mask_a, ids_restore_a, enc_x_mask_a = self.random_masking_audio(
                        x_a, full_x_mask_a, audio_spans=audio_spans)
                else:
                    x_a, mask_a, ids_restore_a, enc_x_mask_a = self.random_masking(
                        x_a, full_x_mask_a)

                enc_mask = torch.cat([enc_x_mask_a, enc_x_mask_v], 1)
                dec_mask = torch.cat([full_x_mask_a, full_x_mask_v], 1)
                x = torch.cat([x_a, x_v], 1)
            if text_embeds is not None:
                enc_mask = torch.cat([text_masks, enc_x_mask_v], 1)
                x = torch.cat([text_embeds, x_v], 1)
                dec_mask = full_x_mask_v

        else:
            if audio is not None and video is not None:
                # 直接拼接音频和视频的掩码，不需要CLS token的掩码
                enc_mask = torch.cat([full_x_mask_a, full_x_mask_v], 1)
                
                # 直接拼接音频和视频特征，不添加CLS token
                x = torch.cat([x_a, x_v], 1)
                
            elif audio is not None:
                enc_mask = full_x_mask_a
                x = x_a

            if text_embeds is not None:
                # 直接拼接文本掩码和视频掩码，不需要CLS token的掩码
                enc_mask = torch.cat([text_masks, full_x_mask_v], 1)
                
                # 直接拼接文本嵌入和视频特征，不添加CLS token
                x = torch.cat([text_embeds, x_v], 1)

        # 进入encoder 
        for blk in self.blocks:
            x = blk(x, enc_mask)
        """
        x = self.transformer(x_a,x_v)
        
        x = self.norm(x)
        # 不加cls token,audio patches(1, 232, 768), patches size[2, 128] ,(1, 1800, 768)
        # 不加cls token,audio patches(1, 196, 768), patches size[16, 16] ,(1, 1764, 768)
        """
        print("\nInput audio shape:", audio.shape)
        print("path size:", self.audio_patch_size)
        print(f"Audio patches shape: {x_a.shape}")
        print(f"Video patches shape: {x_v.shape}")
        print(f"Combined shape: {x.shape}")
        """
        return x

        # 开始decoder
        if mask_visual and use_mae:
            # 将encoder输出降维
            decoder_x = self.decoder_embed(x)

            if audio is not None:
                decoder_x_a = decoder_x[:, :x_a.size(1)]
                decoder_x_a = self.cat_mask(
                    self.mask_token_a, decoder_x_a, ids_restore_a)
                decoder_x_a += self.decoder_freq_embed.repeat(
                    1, decoder_x_a.size(1)//num_freq_patches, 1)
                decoder_x_a += torch.repeat_interleave(self.decoder_pos_embed_a[:, :decoder_x_a.size(
                    1)//num_freq_patches], num_freq_patches, dim=1)[:, :decoder_x_a.size(1)]
                decoder_x_a += self.decoder_type_embed_a
                for i, blk in enumerate(self.decoder_blocks):
                    decoder_x_a = blk(decoder_x_a)
                decoder_x_a = self.decoder_norm(decoder_x_a)
            else:
                decoder_x_a = mask_a = None

            # 添加掩码token
            decoder_x_v = decoder_x[:, -x_v.size(1):]
            decoder_x_v = self.cat_mask(
                self.mask_token_v, decoder_x_v, ids_restore_v)
            # postion embedding
            decoder_x_v += self.decoder_pos_embed_v.repeat(1, t, 1)
            decoder_x_v += torch.repeat_interleave(
                self.decoder_temporal_embed[:, :self.num_frames], frame_patch_len, dim=1)
            decoder_x_v += self.decoder_type_embed_v
            # decoder transformer block
            for blk in self.decoder_blocks:
                decoder_x_v = blk(decoder_x_v)

            decoder_x_v = self.decoder_norm(decoder_x_v)

            return None, decoder_x_a, decoder_x_v, None, mask_a, mask_v

        if text_embeds is not None:
            text_feats = x[:, 1: 1+text_embeds.size(1)]
            return x, None, None, text_feats, None, None
        else:
            return x, None, None, None, None, None


@register_model
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = TVLT(# audio_patch_size=[2, 128]
        patch_size=16, audio_patch_size=[16, 16], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# for32
@register_model
def mae_vit_base_patch32_dec512d8b(**kwargs):
    model = TVLT(
        patch_size=32, audio_patch_size=[16, 16], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def mae_vit_base_patch128_dec512d8b(**kwargs):
    model = TVLT(
        patch_size=16, audio_patch_size=[2, 128], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
