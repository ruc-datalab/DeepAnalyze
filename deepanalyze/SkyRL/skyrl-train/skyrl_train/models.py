# This code is adapted from OpenRLHF and OpenReasonerZero
# https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/models.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/actor.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

from typing import Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import numpy as np
from skyrl_train.distributed.ulysses.utils import (
    ulysses_pad_and_slice_inputs,
    gather_outputs_and_unpad,
)
from skyrl_train.utils.torch_utils import (
    chunked_entropy_from_logits,
    logprobs_from_logits,
)
from flash_attn.bert_padding import pad_input, unpad_input


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        temperature=1.0,
        use_liger_kernel=False,
        sequence_parallel_size=1,
        use_sample_packing: bool = False,
        use_torch_compile: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.sequence_parallel_size = sequence_parallel_size
        if self.sequence_parallel_size > 1:
            logger.info(
                f"Actor model using sequence parallelism with size: {self.sequence_parallel_size}"
            )
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_sample_packing = use_sample_packing
        # packing samples using Flash Attention 2
        if use_sample_packing:
            assert (
                self.use_flash_attention_2
            ), "Flash attention 2 should be used for `use_sample_packing`"

        if isinstance(pretrain_or_model, str):
            attn_implementation = (
                "flash_attention_2" if use_flash_attention_2 else "eager"
            )

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None  # noqa: F841

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

        # TODO (sumanthrh): do the same for `logprobs_from_logits` and test.
        # Credits: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/#efficient-solution
        self.chunked_entropy_from_logits_fn = (
            torch.compile(chunked_entropy_from_logits, dynamic=True)
            if use_torch_compile
            else chunked_entropy_from_logits
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "min_p": kwargs.get("min_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("num_beams", 1) > 1,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(
            sequences, input_ids.size(1), eos_token_id, pad_token_id
        )

    def process_sequences(
        self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id
    ):
        """
        Process generated sequences to create attention masks and action masks.

        Args:
            sequences (torch.Tensor): Generated sequence tensor
            input_len (int): Length of the input sequence
            eos_token_id (int): Token ID for the end-of-sequence token
            pad_token_id (int): Token ID for the padding token

        Returns:
            tuple: A tuple containing three elements:
                - sequences: Original sequence
                - attention_mask: Attention mask indicating valid token positions
                - action_mask: Action mask indicating valid action token positions
        """
        # Create initial attention mask by marking positions that are neither EOS nor padding tokens
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(
            dtype=torch.long
        )
        seq_length = attention_mask.size(1)

        # Find the position of the last valid token in each sequence
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(
            dim=1, keepdim=True
        ).clamp(min=1)

        # Handle cases where EOS tokens might appear in the middle of the prompt (for Llama3 and Qwen2 models)
        # Find the position of the first valid token in each sequence
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # Create position mask
        mask = (
            torch.arange(seq_length)
            .unsqueeze(0)
            .expand(sequences.size(0), -1)
            .to(device=sequences.device)
        )
        # Generate final attention mask, keeping only positions between first and last valid tokens
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(
            dtype=torch.long
        )

        # In reinforcement learning, the state transition is represented as:
        # state_i (current token) + action_i (next token) -> state_i+1 (next token)
        # Generate state sequence from input_len-1 to second-to-last token
        state_seq = sequences[:, input_len - 1 : -1]
        # Generate action mask indicating valid action token positions
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, list[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_output=False,
        compute_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        sequences_fwd = sequences
        position_ids_fwd = position_ids
        attention_mask_fwd = attention_mask
        if self.use_sample_packing:
            with torch.no_grad():
                # Removes padding to get a packed tensor. `unpad_input` expects 3 dimensional tensor so we unsqueeze first
                sequences_fwd, nnz_indices, _, _, _ = unpad_input(
                    sequences.unsqueeze(-1), attention_mask=attention_mask
                )
                # (nnz, 1) -> (1, nnz)
                sequences_fwd = sequences_fwd.transpose(0, 1)
                position_ids_fwd, _, _, _, _ = unpad_input(
                    position_ids.unsqueeze(-1), attention_mask
                )
                # (nnz, 1) -> (1, nnz)
                position_ids_fwd = position_ids_fwd.transpose(0, 1)
                attention_mask_fwd = None  # no attention mask with FA 2

        sequences_rolled = torch.roll(sequences_fwd, shifts=-1, dims=1)
        if self.sequence_parallel_size > 1:
            assert (
                self.use_sample_packing
            ), "sequence packing needs to be enabled for sequence parallelism"
            # don't pass any attention mask for flash attention 2. this will save an all gather.
            attention_mask_fwd = (
                None if self.use_flash_attention_2 else attention_mask_fwd
            )

            # slice for sequence parallelism
            # (bsz, seqlen) -> (bsz, seqlen//sp_size)
            sequences_fwd, position_ids_fwd, attention_mask_fwd, pad_size = (
                ulysses_pad_and_slice_inputs(
                    sequences_fwd,
                    position_ids_fwd,
                    attention_mask_fwd,
                    self.sequence_parallel_size,
                )
            )
            sequences_rolled, _, _, _ = ulysses_pad_and_slice_inputs(
                sequences_rolled, None, None, self.sequence_parallel_size
            )

        # NOTE (sumanthrh): Once we have position_ids, we don't need attention mask with flash attention.
        if self.use_sample_packing and self.use_flash_attention_2:
            # NOTE (sumanthrh): Don't use attention mask. position_ids is enough.
            # Not using attention mask leads to higher perf since flash attention varlen func is enabled
            output = self.model(
                sequences_fwd, attention_mask=None, position_ids=position_ids_fwd
            )
        else:
            output = self.model(
                sequences_fwd,
                attention_mask=attention_mask_fwd,
                position_ids=position_ids_fwd,
            )

        logits_BSV = output["logits"]
        logits_BSV.div_(temperature)

        # NOTE: this is slightly inaccurate with sample packing because last token from nth seq -> first token of n+1th seq loss is added.
        log_probs = logprobs_from_logits(
            logits_BSV,
            sequences_rolled,
            inplace_backward=True,
        )

        # gather output if sp > 1
        if self.sequence_parallel_size > 1:
            log_probs = gather_outputs_and_unpad(
                log_probs.squeeze(0), gather_dim=0, unpad_dim=0, padding_size=pad_size
            )
            # (nnz,) -> (1, nnz)
            log_probs = log_probs.unsqueeze(0)

        if self.use_sample_packing:
            # add padding back - postprocess logprobs to be compatible with original tensor
            batch_size, seqlen = attention_mask.shape
            # (1, nnz-1) -> (batch_size, seqlen). Pad token ID used by flash attention is 0.
            log_probs = pad_input(
                log_probs.transpose(0, 1),
                indices=nnz_indices,
                batch=batch_size,
                seqlen=seqlen,
            ).squeeze(-1)

            # (1, nnz,)
            if compute_entropy:
                # entropy calculation as a metric - we use no grad
                entropy_1N = self.chunked_entropy_from_logits_fn(
                    logits_BSV, requires_grad=False
                )
                if self.sequence_parallel_size > 1:
                    entropy_1N = gather_outputs_and_unpad(
                        entropy_1N.squeeze(0),
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    ).unsqueeze(0)
                entropy_BS = pad_input(
                    entropy_1N.transpose(0, 1),
                    indices=nnz_indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)

                output["entropy"] = entropy_BS

        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                num_actions = np.array(num_actions)
        action_log_probs = log_probs[:, -num_actions - 1 : -1]

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs={"use_reentrant": False}
    ):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(
                sample_length, device=mask.device
            )
    return position_ids


def _get_reward_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="value_head",
    sequence_parallel_size=1,
    use_sample_packing: bool = False,
):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # NOTE (sumanthrh): We initialize base model with random weights first and then use `from_pretrained` to load the weights
            # We ignore warnings about random init
            # TODO (sumanthrh): For efficient initialization, we should use `from_pretrained` directly with meta device
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_model = base_llm_model(config)
            setattr(self, self.base_model_prefix, base_model)

            self.value_head_prefix = value_head_prefix
            setattr(
                self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False)
            )

            self.sequence_parallel_size = sequence_parallel_size
            if self.sequence_parallel_size > 1:
                logger.info(
                    "Reward model using sequence parallelism with size: ",
                    self.sequence_parallel_size,
                )

            self.use_sample_packing = use_sample_packing
            if use_sample_packing:
                assert (
                    config._attn_implementation == "flash_attention_2"
                ), "Flash attention 2 must be used with `use_sample_packing`"

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            pad_size = 0
            input_ids_fwd = input_ids
            position_ids_fwd = position_ids

            if self.use_sample_packing:
                with torch.no_grad():
                    # remove padding. `unpad_input` expects 3 dimensional tensor
                    input_ids_fwd, nnz_indices, _, _, _ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    input_ids_fwd = input_ids_fwd.transpose(0, 1)
                    position_ids_fwd, _, _, _, _ = unpad_input(
                        position_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    position_ids_fwd = position_ids_fwd.transpose(0, 1)
                    attention_mask_fwd = None  # no attention mask with FA2

            if self.sequence_parallel_size > 1:
                # don't pass any attention mask for flash attention 2. this will save an all gather.
                attention_mask_fwd = (
                    None
                    if self.config._attn_implementation == "flash_attention_2"
                    else attention_mask_fwd
                )
                # slice for sequence parallelism
                # (bsz, seqlen) -> (bsz, seqlen//sp_size)
                input_ids_fwd, position_ids_fwd, attention_mask_fwd, pad_size = (
                    ulysses_pad_and_slice_inputs(
                        input_ids_fwd,
                        position_ids_fwd,
                        attention_mask_fwd,
                        self.sequence_parallel_size,
                    )
                )

            if (
                self.sequence_parallel_size > 1
                and self.config._attn_implementation == "flash_attention_2"
            ):
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids_fwd, position_ids=position_ids_fwd
                )
            else:
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids_fwd,
                    attention_mask=attention_mask_fwd,
                    position_ids=position_ids_fwd,
                )
            last_hidden_states_BSH = outputs["last_hidden_state"]
            if self.sequence_parallel_size > 1:
                # (seqlen*bsz//sp_size, 1) -> (seqlen*bsz, 1)
                last_hidden_states_SH = last_hidden_states_BSH.squeeze(0)
                last_hidden_states_SH = gather_outputs_and_unpad(
                    last_hidden_states_SH,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                last_hidden_states_BSH = last_hidden_states_SH.unsqueeze(0)

            # (1, nnz, 1)
            values_BSH = getattr(self, self.value_head_prefix)(last_hidden_states_BSH)

            if self.use_sample_packing:
                # add padding back - postprocess logits to be compatible with original tensors
                batch_size, seqlen = attention_mask.shape
                # (1, nnz, 1) -> (nnz, 1) -> (batch_size, seqlen, 1)
                values_BSH = pad_input(
                    values_BSH.squeeze(0),
                    indices=nnz_indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

            # (batch_size, seqlen, 1) -> (batch_size, seqlen)
            values = values_BSH.squeeze(-1)

            eos_indices = (
                attention_mask.size(1)
                - 1
                - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            )
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                # if mean/std are on cpu due to model cpu offload, move them back to gpu
                if self.mean.device != values.device:
                    self.mean = self.mean.to(values.device)
                    self.std = self.std.to(values.device)
                reward = (reward - self.mean) / (self.std + 1e-8)

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="value_head",
    sequence_parallel_size=1,
    use_sample_packing: bool = False,
):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(
                self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False)
            )

            self.sequence_parallel_size = sequence_parallel_size
            self.use_sample_packing = use_sample_packing
            if use_sample_packing:
                assert (
                    config._attn_implementation == "flash_attention_2"
                ), "Flash attention must be used with sample packing"

            if self.sequence_parallel_size > 1:
                logger.info(
                    "Critic model using sequence parallelism with size: ",
                    self.sequence_parallel_size,
                )

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            input_ids_fwd = input_ids
            position_ids_fwd = position_ids
            attention_mask_fwd = attention_mask

            if self.use_sample_packing:
                with torch.no_grad():
                    # remove padding. `unpad_input` expects 3 dimensional tensor
                    input_ids_fwd, nnz_indices, _, _, _ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    input_ids_fwd = input_ids_fwd.transpose(0, 1)
                    position_ids_fwd, _, _, _, _ = unpad_input(
                        position_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    position_ids_fwd = position_ids_fwd.transpose(0, 1)
                    # don't use attention mask with FA2
                    attention_mask_fwd = None

            if self.sequence_parallel_size > 1:
                assert (
                    self.use_sample_packing
                ), "sample packing must be true for sequence parallelism"
                # don't pass any attention mask for flash attention 2. this will save an all gather.
                attention_mask_fwd = (
                    None
                    if self.config._attn_implementation == "flash_attention_2"
                    else attention_mask
                )
                # slice for sequence parallelism
                # (bsz, seqlen) -> (bsz, seqlen//sp_size)
                input_ids_fwd, position_ids_fwd, attention_mask_fwd, pad_size = (
                    ulysses_pad_and_slice_inputs(
                        input_ids_fwd,
                        position_ids_fwd,
                        attention_mask_fwd,
                        self.sequence_parallel_size,
                    )
                )

            if (
                self.sequence_parallel_size > 1
                and self.config._attn_implementation == "flash_attention_2"
            ):
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids_fwd, position_ids=position_ids_fwd
                )
            else:
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids_fwd,
                    attention_mask=attention_mask_fwd,
                    position_ids=position_ids_fwd,
                )
            last_hidden_states_BSH = outputs["last_hidden_state"]

            if self.sequence_parallel_size > 1:
                last_hidden_states_SH = last_hidden_states_BSH.squeeze(0)
                # (seqlen*bsz//sp_size, 1) -> (seqlen*bsz, 1)
                last_hidden_states_SH = gather_outputs_and_unpad(
                    last_hidden_states_SH,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                last_hidden_states_BSH = last_hidden_states_SH.unsqueeze(0)

            values_BSH = getattr(self, self.value_head_prefix)(last_hidden_states_BSH)

            if self.use_sample_packing:
                # add padding back - postprocess logits to be compatible with original tensors
                batch_size, seqlen = attention_mask.shape
                # (1, nnz, 1) -> (nnz, 1) -> (batch_size, seqlen, 1)
                values_BSH = pad_input(
                    values_BSH.squeeze(0),
                    indices=nnz_indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

            values = values_BSH.squeeze(-1)[:, :-1]

            if num_actions is None:
                assert return_output
                return outputs

            action_values = values[:, -num_actions:]

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="value_head",
    device_map=None,
    sequence_parallel_size=1,
    use_sample_packing: bool = False,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if model_type == "reward":
        config.normalize_reward = normalize_reward
    config._attn_implementation = (
        "flash_attention_2" if use_flash_attention_2 else "eager"
    )

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(
            base_pretrained_class,
            base_class,
            value_head_prefix,
            sequence_parallel_size=sequence_parallel_size,
            use_sample_packing=use_sample_packing,
        )
    else:
        cls_class = _get_critic_model(
            base_pretrained_class,
            base_class,
            value_head_prefix,
            sequence_parallel_size=sequence_parallel_size,
            use_sample_packing=use_sample_packing,
        )

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            import deepspeed

            with deepspeed.zero.GatheredParameters(
                [value_head.weight], modifier_rank=0
            ):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(
                        mean=0.0, std=1 / (config.hidden_size + 1)
                    )
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model
