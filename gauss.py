# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Inference-only Gauss model for SGLang 0.5.x.

This file follows the local ``gemma3_causal.py`` style in this checkout:
HF ``PreTrainedModel`` wrapper, model-level RoPE tensors, and per-layer
``RadixAttention`` with ``sliding_window_size``.

Gauss-specific parts:
* hybrid attention: SWA layers with every fifth layer global by default;
* separate q/k/v projections, because upper layers may compute K/V from
  shared states captured at an earlier layer;
* untied ``lm_head``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import ROPE_INIT_FUNCTIONS, PretrainedConfig, PreTrainedModel

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)

GaussConfig = Any


def get_attention_sliding_window_size(config):
    window = getattr(config, "sliding_window", None)
    if window is None:
        window = getattr(config, "sliding_window_size", 1024)
    if window is None or int(window) <= 0:
        return -1
    # HF-style sliding windows are inclusive; SGLang expects exclusive.
    return int(window) - 1


def _as_int_set(value: Optional[Any]) -> Optional[set[int]]:
    if value is None:
        return None
    if isinstance(value, set):
        return {int(v) for v in value}
    if isinstance(value, (list, tuple)):
        return {int(v) for v in value}
    return {int(value)}


def _is_global_attention_layer(config: GaussConfig, layer_id: int) -> bool:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        layer_type = layer_types[layer_id]
        return layer_type in {
            "full_attention",
            "global_attention",
            "full",
            "global",
        }

    explicit_layers = _as_int_set(getattr(config, "global_attention_layers", None))
    if explicit_layers is None:
        explicit_layers = _as_int_set(getattr(config, "full_attention_layers", None))
    if explicit_layers is not None:
        return layer_id in explicit_layers

    # From your model: 0,1,2,3 SWA and 4 global, repeated every 5 layers.
    pattern = int(getattr(config, "global_attention_pattern", 5))
    offset = int(getattr(config, "global_attention_offset", pattern - 1))
    return layer_id % pattern == offset


def _norm_eps(config: GaussConfig, is_global: bool) -> float:
    if is_global:
        return float(
            getattr(
                config,
                "global_rms_norm_eps",
                getattr(config, "rms_norm_eps", 1e-6),
            )
        )
    return float(
        getattr(
            config,
            "sliding_rms_norm_eps",
            getattr(
                config,
                "sliding_window_rms_norm_eps",
                getattr(config, "rms_norm_eps", 1e-6),
            ),
        )
    )


def _shared_source_for_layer(config: GaussConfig, layer_id: int) -> Optional[int]:
    source_map = getattr(config, "state_shared_source_layers", None)
    if source_map is None:
        source_map = getattr(config, "shared_kv_source_layers", None)

    if isinstance(source_map, dict):
        if layer_id in source_map:
            return int(source_map[layer_id])
        if str(layer_id) in source_map:
            return int(source_map[str(layer_id)])
    elif isinstance(source_map, (list, tuple)):
        if len(source_map) == getattr(config, "num_hidden_layers", len(source_map)):
            source = source_map[layer_id]
            return None if source is None or int(source) < 0 else int(source)

    shared_layers = _as_int_set(getattr(config, "state_shared_layers", None))
    if shared_layers is None:
        shared_layers = _as_int_set(getattr(config, "shared_kv_layers", None))

    default_source = int(
        getattr(
            config,
            "state_shared_source_layer",
            getattr(config, "shared_kv_source_layer", 24),
        )
    )
    if shared_layers is not None:
        return default_source if layer_id in shared_layers else None

    # Default assumption: layers 25+ reuse the output of layer 24 for K/V.
    # If your HF Gauss is "layers 24+ use layer 23", set:
    # state_shared_start_layer=24, state_shared_source_layer=23.
    start_layer = int(
        getattr(
            config,
            "state_shared_start_layer",
            getattr(config, "shared_kv_start_layer", 25),
        )
    )
    return default_source if layer_id >= start_layer else None


def _shared_source_layers(config: GaussConfig) -> set[int]:
    return {
        source
        for layer_id in range(config.num_hidden_layers)
        for source in [_shared_source_for_layer(config, layer_id)]
        if source is not None
    }


class GaussRotaryEmbedding(nn.Module):
    def __init__(self, config: PretrainedConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 32768)
        self.original_max_seq_len = self.max_seq_len_cached
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GaussMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_activation not in ("silu", "swiglu"):
            raise ValueError(
                f"Unsupported activation: {hidden_activation}. "
                "GaussMLP currently supports SiLU/SwiGLU only."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GaussAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads

        if self.total_num_kv_heads < tp_size:
            raise ValueError(
                "GaussAttention uses separate k_proj/v_proj for shared-state KV. "
                "This implementation currently requires tensor_parallel_size <= "
                f"num_key_value_heads ({tp_size} > {self.total_num_kv_heads})."
            )
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.is_sliding = not _is_global_attention_layer(config, layer_id)
        self.sliding_window = (
            get_attention_sliding_window_size(config) if self.is_sliding else None
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        shared_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kv_states = hidden_states if shared_states is None else shared_states

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(kv_states)
        v, _ = self.v_proj(kv_states)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = k.transpose(0, 1).unsqueeze(0)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        attn_output = self.attn(q, k, v, forward_batch=forward_batch)

        if attn_output.dim() == 4 and attn_output.shape[0] == 1:
            attn_output = attn_output.squeeze(0).flatten(-2, -1)

        output, _ = self.o_proj(attn_output)
        return output


class GaussDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.self_attn = GaussAttention(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        hidden_activation = getattr(
            config,
            "hidden_act",
            getattr(config, "hidden_activation", "silu"),
        )
        self.mlp = GaussMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_eps = _norm_eps(config, not self.self_attn.is_sliding)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)
        self.shared_source_layer = _shared_source_for_layer(config, layer_id)
        self.apply_layernorm_to_shared_states = bool(
            getattr(config, "state_shared_apply_layernorm", False)
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        shared_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if shared_states is not None and self.apply_layernorm_to_shared_states:
            shared_states = self.input_layernorm(shared_states)

        position_embeddings = (
            position_embeddings_local
            if self.self_attn.is_sliding
            else position_embeddings_global
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            forward_batch=forward_batch,
            shared_states=shared_states,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class GaussModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.rotary_emb = GaussRotaryEmbedding(config=config)

        local_config = copy.deepcopy(config)
        local_rope_theta = getattr(
            config,
            "rope_local_base_freq",
            getattr(config, "rope_theta", 10000.0),
        )
        local_config.rope_theta = local_rope_theta
        local_config.rope_scaling = getattr(config, "rope_local_scaling", None)
        self.rotary_emb_local = GaussRotaryEmbedding(config=local_config)

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: GaussDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.shared_source_layers = _shared_source_layers(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if positions.dim() == 1:
            positions = positions.view(1, -1)

        position_embeddings_global = self.rotary_emb(hidden_states, positions)
        position_embeddings_local = self.rotary_emb_local(hidden_states, positions)

        shared_states_cache: Dict[int, torch.Tensor] = {}
        for layer in self.layers:
            shared_source = layer.shared_source_layer
            shared_states = None
            if shared_source is not None:
                shared_states = shared_states_cache.get(shared_source)
                if shared_states is None:
                    raise RuntimeError(
                        f"Gauss layer {layer.layer_id} requires shared states "
                        f"from layer {shared_source}, but they were not captured."
                    )

            layer_outputs = layer(
                positions=positions,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                shared_states=shared_states,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if layer.layer_id in self.shared_source_layers:
                shared_states_cache[layer.layer_id] = hidden_states

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GaussForCausalLM(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "model"

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    supported_lora_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.model = GaussModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            **kwargs,
        )

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        if start == 0:
            if input_embeds is None:
                hidden_states = self.model.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds

            if positions.dim() == 1:
                positions = positions.view(1, -1)

            forward_batch.hidden_states = hidden_states
            forward_batch.model_specific_states = {
                "positions": positions,
                "position_embeddings_global": self.model.rotary_emb(
                    hidden_states, positions
                ),
                "position_embeddings_local": self.model.rotary_emb_local(
                    hidden_states, positions
                ),
                "shared_states_cache": {},
            }

        for i in range(start, end):
            layer = self.model.layers[i]
            shared_states_cache = forward_batch.model_specific_states[
                "shared_states_cache"
            ]
            shared_states = None
            if layer.shared_source_layer is not None:
                shared_states = shared_states_cache.get(layer.shared_source_layer)
                if shared_states is None:
                    raise RuntimeError(
                        f"Gauss layer {layer.layer_id} requires shared states "
                        f"from layer {layer.shared_source_layer}, but they were "
                        "not captured in split prefill."
                    )
            layer_output = layer(
                positions=forward_batch.model_specific_states["positions"],
                position_embeddings_global=forward_batch.model_specific_states[
                    "position_embeddings_global"
                ],
                position_embeddings_local=forward_batch.model_specific_states[
                    "position_embeddings_local"
                ],
                hidden_states=forward_batch.hidden_states,
                forward_batch=forward_batch,
                shared_states=shared_states,
            )
            forward_batch.hidden_states = layer_output[0]
            if layer.layer_id in self.model.shared_source_layers:
                shared_states_cache[layer.layer_id] = forward_batch.hidden_states

        if end == self.model.config.num_hidden_layers:
            forward_batch.hidden_states = self.model.norm(forward_batch.hidden_states)
            result = self.logits_processor(
                input_ids,
                forward_batch.hidden_states,
                self.lm_head,
                forward_batch,
            )
        else:
            result = None

        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if name not in params_dict:
                    logger.warning("Parameter %s not found in params_dict", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Gauss3ForCausalLM(GaussForCausalLM):
    pass


EntryClass = [GaussForCausalLM, Gauss3ForCausalLM]
