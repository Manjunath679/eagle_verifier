# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Inference-only Gauss model compatible with HuggingFace-style weights.

This implementation is intentionally close to the Qwen2/Llama SGLang model
files, with two Gauss-specific differences:

* hybrid attention: every fifth layer is global by default, the others use SWA;
* shared-state KV: selected upper layers compute Q from the current hidden state
  while K/V can be projected from a hidden state captured at an earlier layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group, get_pp_indices
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
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
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

GaussConfig = Any


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
            "global",
            "full",
        }

    explicit_global_layers = _as_int_set(
        getattr(config, "global_attention_layers", None)
    )
    if explicit_global_layers is None:
        explicit_global_layers = _as_int_set(
            getattr(config, "full_attention_layers", None)
        )
    if explicit_global_layers is not None:
        return layer_id in explicit_global_layers

    # Gauss default from the notes: 0,1,2,3 are SWA and 4 is global, repeated.
    pattern = int(getattr(config, "global_attention_pattern", 5))
    offset = int(getattr(config, "global_attention_offset", pattern - 1))
    return layer_id % pattern == offset


def _get_sliding_window_size(config: GaussConfig) -> int:
    window = getattr(config, "sliding_window", None)
    if window is None:
        window = getattr(config, "sliding_window_size", 1024)
    if window is None or int(window) <= 0:
        return -1
    # Transformers treats the window as inclusive; SGLang's attention backend
    # expects the exclusive value, like Gemma3/Exaone4 do in-tree.
    return int(window) - 1


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

    # Conservative default for the note "from 24/25th layer, reuse states from
    # 24/23": layers 25+ use layer 24. Override in config if your HF model uses
    # 24+ from 23 instead.
    start_layer = int(
        getattr(
            config,
            "state_shared_start_layer",
            getattr(config, "shared_kv_start_layer", 25),
        )
    )
    if layer_id >= start_layer:
        return default_source
    return None


def _primary_shared_source_layer(config: GaussConfig) -> Optional[int]:
    for layer_id in range(getattr(config, "num_hidden_layers", 0)):
        source = _shared_source_for_layer(config, layer_id)
        if source is not None:
            return source
    return None


class GaussMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
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
        if hidden_act not in ("silu", "swiglu"):
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "GaussMLP currently supports SiLU/SwiGLU only."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch = None) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, forward_batch=forward_batch)
        return x


class GaussAttention(nn.Module):
    def __init__(
        self,
        config: GaussConfig,
        layer_id: int,
        rope_theta: float,
        rope_scaling: Optional[Dict[str, Any]],
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads

        tp_size = get_parallel().tp_size
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads < tp_size:
            raise ValueError(
                "GaussAttention with separate k_proj/v_proj currently requires "
                f"tensor_parallel_size <= num_key_value_heads "
                f"({tp_size} > {self.total_num_kv_heads})."
            )
        assert self.total_num_kv_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        is_global = _is_global_attention_layer(config, layer_id)
        sliding_window_size = -1 if is_global else _get_sliding_window_size(config)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=sliding_window_size,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        shared_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv_states = hidden_states if shared_states is None else shared_states

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(kv_states)
        v, _ = self.v_proj(kv_states)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class GaussDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GaussConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.start_layer = start_layer
        self.shared_source_layer = _shared_source_for_layer(config, layer_id)
        self.apply_layernorm_to_shared_states = bool(
            getattr(config, "state_shared_apply_layernorm", True)
        )

        rope_theta, rope_scaling = get_rope_config(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        self.self_attn = GaussAttention(
            config=config,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = GaussMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(
                config,
                "hidden_act",
                getattr(config, "hidden_activation", "silu"),
            ),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_eps = _norm_eps(config, _is_global_attention_layer(config, layer_id))
        self.input_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        shared_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if shared_states is not None and self.apply_layernorm_to_shared_states:
            shared_states = self.input_layernorm(shared_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            shared_states=shared_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        return hidden_states, residual


class GaussModel(nn.Module):
    def __init__(
        self,
        config: GaussConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: GaussDecoderLayer(
                config=config,
                layer_id=idx,
                start_layer=pp_start_layer,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
            self.norm = RMSNorm(config.hidden_size, eps=norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture: List[int] = []
        self.shared_source_layers = {
            source
            for layer_id in range(config.num_hidden_layers)
            for source in [_shared_source_for_layer(config, layer_id)]
            if source is not None
        }

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
            shared_states_cache: Dict[int, torch.Tensor] = {}
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            shared_states_cache = {}
            shared_states = pp_proxy_tensors.tensors.get("gauss_shared_states")
            shared_source = _primary_shared_source_layer(self.config)
            if shared_states is not None and shared_source is not None:
                shared_states_cache[int(shared_source)] = shared_states

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )

            layer = self.layers[i]
            shared_source = layer.shared_source_layer
            shared_states = None
            if shared_source is not None:
                shared_states = shared_states_cache.get(shared_source)
                if shared_states is None:
                    raise RuntimeError(
                        f"Gauss layer {i} requires shared states from layer "
                        f"{shared_source}, but they were not captured."
                    )

            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                shared_states=shared_states,
            )

            layer_output = (
                hidden_states + residual if residual is not None else hidden_states
            )
            if i in self.shared_source_layers:
                shared_states_cache[i] = layer_output

        if not self.pp_group.is_last_rank:
            proxy = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            primary_source = _primary_shared_source_layer(self.config)
            if primary_source is not None and int(primary_source) in shared_states_cache:
                proxy["gauss_shared_states"] = shared_states_cache[int(primary_source)]
            return PPProxyTensors(proxy)

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling factor")


class GaussForCausalLM(nn.Module):
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

    def __init__(
        self,
        config: GaussConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = GaussModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            return self.pooler(hidden_states, forward_batch)
        return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name == "model.embed_tokens.weight" and getattr(
                self.config, "tie_word_embeddings", False
            ):
                # Gauss checkpoints are expected to have an untied lm_head. If a
                # config accidentally sets tying, preserve SGLang's normal load.
                if "lm_head.weight" in params_dict:
                    param = params_dict["lm_head.weight"]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning("Parameter %s not found in params_dict", name)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class Gauss3ForCausalLM(GaussForCausalLM):
    pass


EntryClass = [GaussForCausalLM, Gauss3ForCausalLM]
