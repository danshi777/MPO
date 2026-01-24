# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override
import deepspeed
import torch.distributed as dist

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps
from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        self.lang_reward = {}

        # hpo hyperparams
        if self.loss_type in ["hpo", "wpo", "hwpo"]:
            mu_en = torch.load(finetuning_args.mu_en_path, map_location="cpu") # [H]
            mu_zh = torch.load(finetuning_args.mu_zh_path, map_location="cpu")
            mu_ja = torch.load(finetuning_args.mu_ja_path, map_location="cpu")
            mu_ko = torch.load(finetuning_args.mu_ko_path, map_location="cpu")
            mu_ar = torch.load(finetuning_args.mu_ar_path, map_location="cpu")
            mu_bn = torch.load(finetuning_args.mu_bn_path, map_location="cpu")
            mu_sw = torch.load(finetuning_args.mu_sw_path, map_location="cpu")
            self.mu_table = torch.stack([mu_en, mu_zh, mu_ja, mu_ko, mu_ar, mu_bn, mu_sw], dim=0)  # [K, H]

            if self.loss_type == "hpo":
                self.margin_coef = finetuning_args.margin_coef
                self.retain_coef = finetuning_args.retain_coef
                logger.info(f"Loss = L_simpo + {self.margin_coef} * L_spatial_margin + {self.retain_coef} * L_retain")
            elif self.loss_type == "wpo":
                self.margin_beta = finetuning_args.margin_beta
                self.retain_coef = finetuning_args.retain_coef
                logger.info(f"Loss = {self.margin_beta} * spatial_margin * L_simpo + {self.retain_coef} * L_retain")
            else: # "hwpo"
                self.margin_coef = finetuning_args.margin_coef
                self.retain_coef = finetuning_args.retain_coef
                self.margin_beta = finetuning_args.margin_beta
                logger.info(f"Loss = {self.margin_beta} * spatial_margin * L_simpo + {self.margin_coef} * L_spatial_margin + {self.retain_coef} * L_retain")


        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def mpo_loss(
        self, 
        source_chosen_logps: "torch.Tensor", 
        source_rejected_logps: "torch.Tensor", 
        target_chosen_logps: "torch.Tensor", 
        target_rejected_logps: "torch.Tensor"
    ) -> "torch.Tensor":
        r"""
        Computes MPO loss for batched log probabilities of the policy model.
        """
        target_gap = target_chosen_logps - target_rejected_logps # shape: [B,]
        
        source_gap = (source_chosen_logps - source_rejected_logps) # / self.beta # shape: [B,]

        mpo_loss = nn.MSELoss()(self.beta * target_gap, source_gap) # nn.MSELoss() 默认 reduction="mean"，因此输出标量
        return mpo_loss

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """
        x:   [N, L, H]
        mask:[N, L]  (bool)
        return: [N, H]
        """
        mask_f = mask.to(x.dtype).unsqueeze(-1)          # [N, L, 1]
        denom = mask_f.sum(dim=dim).clamp_min(1.0)       # [N, 1]
        return (x * mask_f).sum(dim=dim) / denom         # [N, H]
        # x * mask_f：将无效位置的隐藏状态清零, shape [N, L, H]
        # sum(dim=dim)：对有效位置的隐藏状态求和, shape [N, H]
        # / denom：除以有效 token 数量，得到 masked mean, shape [N, H]

    def extract_prompt_repr(
        self,
        last_hidden: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """
        last_hidden:    [N, L, H]  (all_outputs.hidden_states[-1])
        labels:         [N, L]
        attention_mask: [N, L]
        return: prompt_repr [N, H]  仅对 prompt token 做 masked mean
        """
        # prompt tokens 通常在 labels 中是 IGNORE_INDEX；padding 由 attention_mask=0 指示
        prompt_mask = (labels == ignore_index) & (attention_mask == 1)  # [N, L]
        return self.masked_mean(last_hidden, prompt_mask, dim=1)        # [N, H]

    def _filter_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        allowed = {"input_ids", "attention_mask", "labels", "position_ids", "inputs_embeds", "images", "videos"}
        return {k: v for k, v in batch.items() if k in allowed}
        
    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        
        该函数的目标是：
        对一个“拼接后的 batch”（chosen+rejected）做一次 forward，得到：
            1.chosen / rejected 的 平均 token log-prob（logps）
            2.chosen / rejected 的 logits（用于某些 loss/辅助项）
            3.最后一层 hidden states 的句级表示（此处是对序列维做 mean pooling）
            4.额外返回一个 “长度归一化后的 chosen_logps”（更准确说是：每 token 的平均 logp 再除以长度，这点需要注意）。
        """
        # filter "source_language_id" and "target_language_id"
        batch = self._filter_model_inputs(batch)
        # batch为：
            # input_ids: shape [2B, L], 2B：拼接后的 batch（B 个 chosen + B 个 rejected）; L：padding 到同一长度后的序列长度
            # attention_mask: shape [2B, L]
            # labels: shape [2B, L]
            # Causal LM 常用：prompt 部分 label 为 -100（忽略），response 部分为 token id（用于计算 logp）
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        all_outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=True) # 一次 forward 同时算 chosen+rejected
        # all_outputs.logits: shape [2B, L, V(vocabe_size)]
        # all_outputs.hidden_states: 通常是长度为 num_layers+1 的 tuple，每一项 shape [2B, L, H(hidden_size)]（embedding 输出 + 每层输出）
        # use_cache=False：训练时禁用 KV cache，避免显存与梯度问题
        # output_hidden_states=True：为了拿最后层 hidden
        all_logits: "torch.Tensor" = all_outputs.logits.to(torch.float32) # [2B, L, V]
        all_hidden_states: "torch.Tensor" = all_outputs.hidden_states[-1] # [2B, L, H]

        # >>> 新增：prompt-only repr（注意：用原始 labels/attention_mask，不做 shift）
        all_prompt_repr = self.extract_prompt_repr(
            last_hidden=all_hidden_states,
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            ignore_index=IGNORE_INDEX,
        )  # [2B, H]

        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"]) # shape 都是[2B,]
        all_logps = all_logps / valid_length # 做长度归一化，shape [2B], 这对应 SimPO 等方法常用的“平均 token logp”，而不是总和 logp

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0) # shape 都是[B,]
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0) # shape 都是[B, L, V]
        chosen_prompt_repr, rejected_prompt_repr = all_prompt_repr.split(batch_size, dim=0)  # shape 都是[B, H]

        chosen_length, _ = valid_length.split(batch_size, dim=0) # shape 都是[B,]

        # return chosen_logps, rejected_logps, chosen_prompt_repr, rejected_prompt_repr, all_hidden_states, chosen_logits, rejected_logits, chosen_logps / chosen_length
        # return chosen_logps, rejected_logps, chosen_prompt_repr, rejected_prompt_repr, all_hidden_states
        return chosen_logps, rejected_logps, chosen_prompt_repr, rejected_prompt_repr, all_prompt_repr

    @override
    def compute_reference_reprs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        # print("use_ref_model =", self.finetuning_args.use_ref_model)
        if not self.finetuning_args.use_ref_model:
            return None, None

        # 两种情况：
        # self.ref_model is None：reference 就是 policy model 的底座，但通过 disable_adapter() 关闭 LoRA/adapter（即 reference = base，不带 adapter 的版本）。
        # self.ref_model 存在：用一个独立冻结的 reference 模型。
        # print("self.ref_model:", self.ref_model) # True
        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            # reference_chosen_logps, reference_rejected_logps, ref_chosen_prompt_repr, ref_rejected_prompt_repr, reference_all_hidden_states, *_ = self.concatenated_forward(ref_model, batch)
            reference_chosen_logps, reference_rejected_logps, ref_chosen_prompt_repr, ref_rejected_prompt_repr, reference_all_hidden_states = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps, ref_chosen_prompt_repr, ref_rejected_prompt_repr, reference_all_hidden_states


    def l2_dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: [B, H]
        b: [H] or [B, H]
        return: [B]
        """
        if b.dim() == 1:
            b = b.unsqueeze(0)  # [1, H]
        return torch.norm(a - b, p=2, dim=-1)  # [B] 
    
    def rms_dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: [B, H]
        b: [H] or [B, H]
        return: [B]
        """
        if b.dim() == 1:
            b = b.unsqueeze(0)  # [1, H]
        return torch.sqrt(((a - b) ** 2).mean(dim=-1) + 1e-8)  # [B]


    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss
    

    def spatial_margin_loss(
        self, 
        source_chosen_reprs: "torch.Tensor", # [B, H]
        source_rejected_reprs: "torch.Tensor", # [B, H]
        target_chosen_reprs: "torch.Tensor", # [B, H]
        target_rejected_reprs: "torch.Tensor", # [B, H]
        source_lang_ids, # [2B, H]
        target_lang_ids, # [2B, H]
    ) -> "torch.Tensor":
        r"""
        Computes HPO loss for batched representations of the policy model.
        """
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     print("Computing Hpo Loss......")
        #     print("-------------------- chosen --------------------------")
        #     print(source_chosen_reprs)
        #     print("-------------------- rejected --------------------------")
        #     print(source_rejected_reprs)
        # 已确认source_chosen_reprs和source_rejected_reprs是一样的

        mu_table = self.mu_table.to(self.accelerator.device)  # [K, H]

        batch_size = source_lang_ids.size(0) // 2
        source_lang_ids_b = source_lang_ids[:batch_size]
        target_lang_ids_b = target_lang_ids[:batch_size]
        mu_harmless_sources = mu_table[source_lang_ids_b]  # [B, H]  <<< 核心：按样本选 μ
        mu_harmless_targets = mu_table[target_lang_ids_b] 

        # source_margin = self.l2_dist(source_chosen_reprs, mu_harmless_sources) # [B], no grad
        # target_margin = self.l2_dist(target_chosen_reprs, mu_harmless_targets) # [B], grad
        source_margin = self.rms_dist(source_chosen_reprs, mu_harmless_sources) # [B], no grad
        target_margin = self.rms_dist(target_chosen_reprs, mu_harmless_targets) # [B], grad
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     print("en_margin:", source_margin)
        #     print("target_margin:", target_margin)

        margin_violation = torch.relu(source_margin - target_margin)      # [B], relu相当于max(0,x)，只当source_margin > target_margin 时才拉近
        # margin_violation = source_margin / target_margin

        # spatial_margin_loss = (margin_violation ** 2).mean() # scalar
        spatial_margin_loss = margin_violation  # [B]
        return spatial_margin_loss


    def compute_preference_loss(
        self,
        policy_outputs: "dict",
        reference_outputs: "dict",
        source_lang_ids,
        target_lang_ids,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        # for spatial_margin_loss
        if self.finetuning_args.use_ref_model:
            source_chosen_reprs = reference_outputs["source"][2].detach()
            source_rejected_reprs = reference_outputs["source"][3].detach()
        target_chosen_reprs = policy_outputs["target"][2]
        target_rejected_reprs = policy_outputs["target"][3]
        
        # for simpo_loss
        policy_chosen_logps = policy_outputs["target"][0]
        policy_rejected_logps = policy_outputs["target"][1]

        simpo_loss = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        
        if self.loss_type == "simpo":
            losses = simpo_loss
        elif self.loss_type == "hpo":
            spatial_margin_loss = self.spatial_margin_loss(
                source_chosen_reprs, 
                source_rejected_reprs, 
                target_chosen_reprs, 
                target_rejected_reprs,
                source_lang_ids,
                target_lang_ids,
                ) # 结果：标量
            # policy_outputs["source"][4]: 来自 concatenated_forward 的 all_hidden_states，shape [2B, L, H]
            # policy_outputs["source"][4]: 来自 concatenated_forward 的 all_prompt_repr，是prompt-only的表示，shape [2B, H]
            en_retain_loss = nn.MSELoss()(
                policy_outputs["source"][4].to(self.accelerator.device), 
                reference_outputs["source"][4].to(self.accelerator.device),
            )
            # en_retain_loss：保持英语的hiddenstates和ref model尽量不变
            # 它在对齐 chosen+rejected 全部样本的 pooled hidden（2B 个），而不是只对齐 chosen 或 prompt-only。
            # 由于 all_last_hidden_states 是 .mean(dim=1) 得到的句向量，会被 padding 强烈影响（如果 batch 内长度差大）。
            # 如果你 intended 的 retain 是“保持英语语义/能力”，更常见做法是：
                # 只对齐 prompt 表示，或
                # 对齐 response-free 的 hidden，或
                # 用 masked mean（attention mask）做 pooling
            losses = simpo_loss + self.margin_coef * spatial_margin_loss + en_retain_loss
        elif self.loss_type == "wpo":
            spatial_margin_loss = self.spatial_margin_loss(
                source_chosen_reprs, 
                source_rejected_reprs, 
                target_chosen_reprs, 
                target_rejected_reprs,
                source_lang_ids,
                target_lang_ids,
                ) # 结果：标量
            en_retain_loss = nn.MSELoss()(
                policy_outputs["source"][4].to(self.accelerator.device), 
                reference_outputs["source"][4].to(self.accelerator.device),
            )
            losses = self.margin_beta * spatial_margin_loss.detach() * simpo_loss
            # losses = self.margin_beta * 1.0 * simpo_loss + en_retain_loss
            # losses = self.margin_beta * 1.0 * simpo_loss
        elif self.loss_type == "hwpo":
            spatial_margin_loss = self.spatial_margin_loss(
                source_chosen_reprs, 
                source_rejected_reprs, 
                target_chosen_reprs, 
                target_rejected_reprs,
                source_lang_ids,
                target_lang_ids,
                ) # 结果：标量
            en_retain_loss = nn.MSELoss()(
                policy_outputs["source"][4].to(self.accelerator.device), 
                reference_outputs["source"][4].to(self.accelerator.device),
            )
            losses = self.margin_beta * spatial_margin_loss.detach() * simpo_loss + self.margin_coef * spatial_margin_loss + en_retain_loss
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

        chosen_rewards = policy_chosen_logps.to(self.accelerator.device).detach() # rewards就等于logps. 注意这里 .detach()：不让这些指标张量参与梯度（只用于 logging）。
        rejected_rewards = policy_rejected_logps.to(self.accelerator.device).detach()

        if self.loss_type == "simpo":
            return losses, chosen_rewards, rejected_rewards
        else: # hpo
            return losses, simpo_loss, spatial_margin_loss, en_retain_loss, chosen_rewards, rejected_rewards
    

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        batch[0]: source language（英语/主导语言）的拼接样本 dict，包含 input_ids/labels/...，shape 如前：[2B, L]。
        batch[1]: target language（低资源语言）的拼接样本 dict。
        
        Computes the MPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        
        policy_outputs = {}
        policy_outputs["source"] = self.concatenated_forward(model, batch[0])
        policy_outputs["target"] = self.concatenated_forward(model, batch[1])

        reference_outputs = {}
        reference_outputs['source'] = self.compute_reference_reprs(model, batch[0])
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     print("batch[0]:", batch[0]['input_ids'][0])
    
        source_lang_ids = batch[0]["source_language_id"].to(self.accelerator.device)  # [2B]
        target_lang_ids = batch[1]["target_language_id"].to(self.accelerator.device)  # [2B]
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     print("source_lang_ids: ", source_lang_ids)
        #     print("target_lang_ids: ", target_lang_ids)


        # print("reference_outputs[source]:", reference_outputs['source'])
        if self.loss_type == "simpo":
            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_outputs,
                reference_outputs,
                source_lang_ids,
                target_lang_ids,
            )
        else: # hpo
            losses, simpo_loss, spatial_margin_loss, en_retain_loss, chosen_rewards, rejected_rewards= self.compute_preference_loss(
                policy_outputs,
                reference_outputs,
                source_lang_ids,
                target_lang_ids,
            )
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        
        if self.loss_type in ["hpo", "wpo", "hwpo"]:
            metrics["{}simpo_loss".format(prefix)] = simpo_loss.detach().float().mean().cpu().item()
            metrics["{}spatial_margin_loss".format(prefix)] = spatial_margin_loss.detach().float().mean().cpu().item()
            metrics["{}en_retain_loss".format(prefix)] = en_retain_loss.detach().float().mean().cpu().item()

        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()

        return losses.mean(), metrics
