from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .cnets import LlamaDecoderLayeremb, LlamaRMSNorm
except ImportError:
    from cnets import LlamaDecoderLayeremb, LlamaRMSNorm


@dataclass
class RefinedDraftTree:
    draft_tokens: Tensor
    retrieve_indices: Tensor
    tree_mask: Tensor
    tree_position_ids: Tensor
    node_scores: Tensor
    kept_indices: Tensor


class DraftBranchRefiner(nn.Module):
    """
    One-pass second draft stage for EAGLE-3 trees.

    The first EAGLE layer still proposes a wide tree. This module reuses the
    same decoder-layer style over that tree once, scores the proposed nodes, and
    keeps only the strongest children per already-kept parent before target
    verification. The target model remains the final verifier, so greedy
    decoding stays lossless; this stage only changes which draft branches are
    worth spending target-model compute on.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = getattr(config, "branch_refiner_num_hidden_layers", 1)
        self.mode = getattr(config, "branch_refiner_mode", "score")
        self.keep_per_depth = getattr(config, "branch_refiner_keep_per_depth", 2)
        self.keep_schedule = _normalize_int_schedule(
            getattr(config, "branch_refiner_keep_schedule", None)
        )
        self.max_depth = getattr(config, "branch_refiner_max_depth", None)
        self.max_tokens = getattr(config, "branch_refiner_max_tokens", None)
        self.draft_score_weight = getattr(config, "branch_refiner_draft_score_weight", 1.0)
        self.token_score_weight = getattr(config, "branch_refiner_token_score_weight", 1.0)
        self.branch_score_weight = getattr(config, "branch_refiner_branch_score_weight", 1.0)
        dropout = getattr(config, "branch_refiner_dropout", 0.0)

        self.score_proj = nn.Linear(1, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            LlamaDecoderLayeremb(config) for _ in range(self.num_layers)
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.keep_head = nn.Linear(config.hidden_size, 1, bias=False)
        nn.init.zeros_(self.score_proj.weight)
        nn.init.zeros_(self.keep_head.weight)

    def forward(
        self,
        input_embeds: Tensor,
        context_states: Tensor,
        tree_mask: Tensor,
        tree_position_ids: Tensor,
        draft_scores: Tensor | None = None,
        lm_head: nn.Module | None = None,
        draft_token_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if tree_position_ids.dim() == 1:
            position_ids = tree_position_ids.unsqueeze(0).expand(
                input_embeds.shape[0], -1
            )
        else:
            position_ids = tree_position_ids

        hidden_states = context_states
        if draft_scores is not None:
            hidden_states = hidden_states + self.score_proj(draft_scores.unsqueeze(-1))
        hidden_states = self.dropout(hidden_states)

        attention_mask = self._tree_attention_mask(tree_mask, hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                input_emb=input_embeds.to(hidden_states.dtype),
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )[0]

        hidden_states = self.norm(hidden_states)
        branch_scores = self.keep_head(hidden_states).squeeze(-1)
        node_scores = self.branch_score_weight * branch_scores
        if draft_scores is not None:
            node_scores = node_scores + self.draft_score_weight * draft_scores
        token_logits = lm_head(hidden_states) if lm_head is not None else None
        if self.mode == "rescore":
            if token_logits is None or draft_token_ids is None:
                raise ValueError("rescore mode requires a shared lm_head and draft_token_ids")
            token_scores = self._child_token_scores(
                token_logits=token_logits,
                draft_token_ids=draft_token_ids,
                tree_mask=tree_mask,
                position_ids=position_ids,
            )
            node_scores = node_scores + self.token_score_weight * token_scores
        elif self.mode != "score":
            raise ValueError(f"unknown branch_refiner_mode: {self.mode}")
        return node_scores, token_logits, hidden_states

    @torch.no_grad()
    def refine_tree(
        self,
        embed_tokens: nn.Embedding,
        lm_head: nn.Module,
        context_state: Tensor,
        draft_tokens: Tensor,
        tree_mask: Tensor,
        tree_position_ids: Tensor,
        draft_scores: Tensor | None = None,
        draft_token_ids: Tensor | None = None,
        keep_per_depth: int | None = None,
        max_depth: int | None = None,
        max_tokens: int | None = None,
        sort_paths: bool = False,
    ) -> RefinedDraftTree:
        if draft_tokens.shape[0] != 1:
            raise NotImplementedError("DraftBranchRefiner currently supports batch_size=1")

        input_embeds = embed_tokens(draft_tokens.to(embed_tokens.weight.device))
        if context_state.dim() == 2:
            context_states = context_state[:, None].expand(
                draft_tokens.shape[0], draft_tokens.shape[1], -1
            )
        else:
            context_states = context_state
        context_states = context_states.to(input_embeds.device)
        if draft_scores is not None:
            draft_scores = draft_scores.to(input_embeds.device)
        if draft_token_ids is not None:
            draft_token_ids = draft_token_ids.to(input_embeds.device)

        node_scores, _, _ = self(
            input_embeds=input_embeds,
            context_states=context_states,
            tree_mask=tree_mask.to(input_embeds.device),
            tree_position_ids=tree_position_ids.to(input_embeds.device),
            draft_scores=draft_scores,
            lm_head=lm_head,
            draft_token_ids=draft_token_ids,
        )
        return prune_branch_tree(
            draft_tokens=draft_tokens,
            tree_mask=tree_mask,
            tree_position_ids=tree_position_ids,
            node_scores=node_scores,
            keep_per_depth=(
                keep_per_depth
                or self.keep_schedule
                or self.keep_per_depth
            ),
            max_depth=self.max_depth if max_depth is None else max_depth,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            sort_paths=sort_paths,
        )

    def _tree_attention_mask(self, tree_mask: Tensor, hidden_states: Tensor) -> Tensor:
        if tree_mask.dim() == 2:
            tree_mask = tree_mask[None, None]
        elif tree_mask.dim() == 3:
            tree_mask = tree_mask[:, None]
        if tree_mask.shape[0] == 1 and hidden_states.shape[0] > 1:
            tree_mask = tree_mask.expand(hidden_states.shape[0], -1, -1, -1)

        attention_mask = torch.zeros(
            tree_mask.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return attention_mask.masked_fill(
            tree_mask.to(hidden_states.device).eq(0),
            torch.finfo(hidden_states.dtype).min,
        )

    def _child_token_scores(
        self,
        token_logits: Tensor,
        draft_token_ids: Tensor,
        tree_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        if token_logits.shape[0] != 1:
            raise NotImplementedError("tree rescoring currently supports batch_size=1")

        tree_2d = _as_tree_2d(tree_mask).to(token_logits.device).bool()
        pos_1d = position_ids[0].to(token_logits.device).long()
        parent_ids = _parent_indices(tree_2d, pos_1d)
        token_logp = F.log_softmax(token_logits.float(), dim=-1)
        child_scores = token_logits.new_zeros(token_logits.shape[:2])
        for child_idx in range(1, token_logits.shape[1]):
            parent_idx = int(parent_ids[child_idx].item())
            child_token = draft_token_ids[:, child_idx : child_idx + 1].to(
                token_logits.device
            )
            child_scores[:, child_idx] = token_logp[:, parent_idx].gather(
                dim=-1,
                index=child_token,
            ).squeeze(-1)
        return child_scores

    def compute_verifier_loss(
        self,
        token_logits: Tensor,
        target_logits: Tensor,
        draft_token_ids: Tensor,
        tree_mask: Tensor,
        tree_position_ids: Tensor,
        keep_labels: Tensor | None = None,
        node_scores: Tensor | None = None,
        node_mask: Tensor | None = None,
        token_labels: Tensor | None = None,
        parent_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return draft_verifier_loss(
            token_logits=token_logits,
            target_logits=target_logits,
            draft_token_ids=draft_token_ids,
            tree_mask=tree_mask,
            tree_position_ids=tree_position_ids,
            keep_labels=keep_labels,
            node_scores=node_scores,
            node_mask=node_mask,
            token_labels=token_labels,
            parent_indices=parent_indices,
            distill_weight=getattr(self.config, "branch_refiner_distill_weight", 1.0),
            token_weight=getattr(self.config, "branch_refiner_token_ce_weight", 0.5),
            edge_weight=getattr(self.config, "branch_refiner_edge_weight", 1.0),
            branch_weight=getattr(self.config, "branch_refiner_branch_bce_weight", 0.0),
            temperature=getattr(self.config, "branch_refiner_temperature", 1.0),
            offpolicy_distill_weight=getattr(
                self.config, "branch_refiner_offpolicy_distill_weight", 0.1
            ),
            offpolicy_token_weight=getattr(
                self.config, "branch_refiner_offpolicy_token_weight", 0.1
            ),
            rejected_edge_weight=getattr(
                self.config, "branch_refiner_rejected_edge_weight", 1.0
            ),
            dead_edge_weight=getattr(self.config, "branch_refiner_dead_edge_weight", 0.25),
        )


@torch.no_grad()
def prune_branch_tree(
    draft_tokens: Tensor,
    tree_mask: Tensor,
    tree_position_ids: Tensor,
    node_scores: Tensor,
    keep_per_depth: int | list[int] | tuple[int, ...] = 2,
    max_depth: int | None = None,
    max_tokens: int | None = None,
    sort_paths: bool = False,
) -> RefinedDraftTree:
    keep_schedule = _normalize_int_schedule(keep_per_depth)
    if not keep_schedule:
        raise ValueError("keep_per_depth must provide at least one value")
    if draft_tokens.shape[0] != 1:
        raise NotImplementedError("branch tree pruning currently supports batch_size=1")

    device = draft_tokens.device
    tree_2d = _as_tree_2d(tree_mask).to(device).bool()
    position_ids = tree_position_ids.to(device).long().view(-1)
    scores = node_scores.to(device).view(-1)
    parent_ids = _parent_indices(tree_2d, position_ids)

    observed_depth = int(position_ids.max().item())
    if max_depth is None:
        max_depth = observed_depth
    else:
        max_depth = min(max_depth, observed_depth)

    max_nodes = None if max_tokens is None else max_tokens + 1
    kept: list[int] = [0]
    frontier: list[int] = [0]
    for depth in range(1, max_depth + 1):
        next_frontier: list[int] = []
        depth_keep = _schedule_value(keep_schedule, depth - 1)
        for parent in frontier:
            child_mask = (parent_ids == parent) & (position_ids == depth)
            children = torch.nonzero(child_mask, as_tuple=True)[0]
            if children.numel() == 0:
                continue
            top_count = min(depth_keep, children.numel())
            top_local = torch.topk(scores[children], top_count).indices
            chosen = children[top_local]
            next_frontier.extend(chosen.tolist())

        if not next_frontier:
            break
        if max_nodes is not None:
            remaining = max_nodes - len(kept)
            if remaining <= 0:
                break
            if len(next_frontier) > remaining:
                next_tensor = torch.tensor(next_frontier, device=device, dtype=torch.long)
                top_local = torch.topk(scores[next_tensor], remaining).indices
                next_frontier = next_tensor[top_local].tolist()
        next_frontier = sorted(next_frontier)
        kept.extend(next_frontier)
        frontier = next_frontier

    if len(kept) == 1 and draft_tokens.shape[1] > 1:
        fallback = torch.argmax(scores[1:]).item() + 1
        kept.append(fallback)
        kept.extend(_ancestor_chain(parent_ids, fallback))
        kept = sorted(set(kept), key=lambda idx: (int(position_ids[idx].item()), idx))
        if kept[0] != 0:
            kept.insert(0, 0)

    kept = _dedupe_topological(kept, position_ids)
    kept_indices = torch.tensor(kept, device=device, dtype=torch.long)
    new_tokens = draft_tokens.index_select(1, kept_indices)
    new_tree = tree_2d.index_select(0, kept_indices).index_select(1, kept_indices)
    new_position_ids = new_tree.long().sum(dim=1) - 1
    retrieve_indices = _retrieve_indices_from_tree(
        new_tree,
        new_position_ids,
        sort_paths=sort_paths,
    )

    return RefinedDraftTree(
        draft_tokens=new_tokens,
        retrieve_indices=retrieve_indices.to(device),
        tree_mask=new_tree.float()[None, None],
        tree_position_ids=new_position_ids.to(device),
        node_scores=scores.index_select(0, kept_indices),
        kept_indices=kept_indices,
    )


def _as_tree_2d(tree_mask: Tensor) -> Tensor:
    if tree_mask.dim() == 4:
        return tree_mask[0, 0]
    if tree_mask.dim() == 3:
        return tree_mask[0]
    if tree_mask.dim() == 2:
        return tree_mask
    raise ValueError(f"unexpected tree_mask shape: {tuple(tree_mask.shape)}")


def _as_tree_batch(tree_mask: Tensor, batch_size: int) -> Tensor:
    if tree_mask.dim() == 4:
        tree = tree_mask[:, 0]
    elif tree_mask.dim() == 3:
        tree = tree_mask
    elif tree_mask.dim() == 2:
        tree = tree_mask[None]
    else:
        raise ValueError(f"unexpected tree_mask shape: {tuple(tree_mask.shape)}")
    if tree.shape[0] == 1 and batch_size > 1:
        tree = tree.expand(batch_size, -1, -1)
    if tree.shape[0] != batch_size:
        raise ValueError(
            f"tree batch {tree.shape[0]} does not match logits batch {batch_size}"
        )
    return tree


def _as_position_batch(position_ids: Tensor, batch_size: int) -> Tensor:
    if position_ids.dim() == 1:
        pos = position_ids[None]
    elif position_ids.dim() == 2:
        pos = position_ids
    else:
        raise ValueError(f"unexpected position_ids shape: {tuple(position_ids.shape)}")
    if pos.shape[0] == 1 and batch_size > 1:
        pos = pos.expand(batch_size, -1)
    if pos.shape[0] != batch_size:
        raise ValueError(
            f"position batch {pos.shape[0]} does not match logits batch {batch_size}"
        )
    return pos


def tree_parent_indices(
    tree_mask: Tensor,
    position_ids: Tensor,
    batch_size: int | None = None,
) -> Tensor:
    if batch_size is None:
        if tree_mask.dim() >= 3:
            batch_size = tree_mask.shape[0]
        elif position_ids.dim() == 2:
            batch_size = position_ids.shape[0]
        else:
            batch_size = 1
    tree = _as_tree_batch(tree_mask, batch_size).bool()
    pos = _as_position_batch(position_ids, batch_size).to(tree.device).long()
    parent_ids = torch.full_like(pos, -1)
    for batch_idx in range(batch_size):
        parent_ids[batch_idx] = _parent_indices(tree[batch_idx], pos[batch_idx])
    return parent_ids


def _normalize_int_schedule(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [max(1, value)]
    return [max(1, int(item)) for item in value]


def _schedule_value(schedule: list[int], index: int) -> int:
    if index < len(schedule):
        return schedule[index]
    return schedule[-1]


def _parent_indices(tree_mask: Tensor, position_ids: Tensor) -> Tensor:
    parent_ids = torch.full_like(position_ids, -1)
    for idx in range(1, position_ids.numel()):
        parent_depth = position_ids[idx] - 1
        ancestors = torch.nonzero(
            tree_mask[idx] & position_ids.eq(parent_depth),
            as_tuple=True,
        )[0]
        parent_ids[idx] = ancestors[-1] if ancestors.numel() else 0
    return parent_ids


def _ancestor_chain(parent_ids: Tensor, node_idx: int) -> list[int]:
    ancestors: list[int] = []
    parent = int(parent_ids[node_idx].item())
    while parent > 0:
        ancestors.append(parent)
        parent = int(parent_ids[parent].item())
    return ancestors


def _dedupe_topological(indices: list[int], position_ids: Tensor) -> list[int]:
    deduped = sorted(set(indices), key=lambda idx: (int(position_ids[idx].item()), idx))
    if not deduped or deduped[0] != 0:
        deduped.insert(0, 0)
    return deduped


def _retrieve_indices_from_tree(
    tree_mask: Tensor,
    position_ids: Tensor,
    sort_paths: bool,
) -> Tensor:
    parent_ids = _parent_indices(tree_mask, position_ids)
    node_count = position_ids.numel()
    parent_set = set(parent_ids[1:].tolist())
    leaves = [idx for idx in range(node_count) if idx not in parent_set]
    max_depth = int(position_ids.max().item()) + 1
    rows: list[list[int]] = []
    for leaf in leaves:
        row = [-1] * max_depth
        cursor = leaf
        while cursor >= 0:
            depth = int(position_ids[cursor].item())
            row[depth] = cursor
            if cursor == 0:
                break
            cursor = int(parent_ids[cursor].item())
        rows.append(row)

    if sort_paths:
        max_item = node_count + 5

        def sort_key(row: list[int]) -> list[int]:
            return [idx if idx >= 0 else max_item for idx in row]

        rows = sorted(rows, key=sort_key)

    return torch.tensor(rows, dtype=torch.long, device=tree_mask.device)


def build_accept_labels_from_target_logits(
    target_logits: Tensor,
    draft_token_ids: Tensor,
    tree_mask: Tensor,
    tree_position_ids: Tensor,
    node_mask: Tensor | None = None,
    parent_indices: Tensor | None = None,
) -> dict[str, Tensor]:
    """
    Build greedy target accept labels for a draft tree.

    `target_logits[b, parent]` is the target distribution for the next token
    from that parent context. A child node is path-accepted only when its token
    matches the target argmax at the parent and every ancestor is accepted.
    """
    if target_logits.dim() != 3:
        raise ValueError("target_logits must have shape [batch, nodes, vocab]")
    batch_size, node_count, _ = target_logits.shape
    device = target_logits.device
    draft_token_ids = _ensure_batch_nodes(draft_token_ids, batch_size, node_count).to(device)
    valid_nodes = _valid_node_mask(node_mask, batch_size, node_count, device)
    if parent_indices is None:
        parent_indices = tree_parent_indices(
            tree_mask=tree_mask,
            position_ids=tree_position_ids,
            batch_size=batch_size,
        ).to(device)
    else:
        parent_indices = _ensure_batch_nodes(parent_indices, batch_size, node_count).to(device)

    target_argmax = target_logits.argmax(dim=-1)
    keep_labels = torch.zeros(
        batch_size, node_count, dtype=torch.bool, device=device
    )
    local_accept = torch.zeros_like(keep_labels)
    keep_labels[:, 0] = valid_nodes[:, 0]
    local_accept[:, 0] = valid_nodes[:, 0]

    position_ids = _as_position_batch(tree_position_ids, batch_size).to(device)
    max_depth = int(position_ids[valid_nodes].max().item()) if valid_nodes.any() else 0
    for depth in range(1, max_depth + 1):
        depth_nodes = torch.nonzero(
            (position_ids == depth) & valid_nodes,
            as_tuple=False,
        )
        for batch_idx, node_idx in depth_nodes.tolist():
            parent_idx = int(parent_indices[batch_idx, node_idx].item())
            if parent_idx < 0:
                continue
            parent_target = target_argmax[batch_idx, parent_idx]
            child_token = draft_token_ids[batch_idx, node_idx]
            local = child_token == parent_target
            local_accept[batch_idx, node_idx] = local
            keep_labels[batch_idx, node_idx] = (
                bool(keep_labels[batch_idx, parent_idx].item()) and bool(local.item())
            )

    return {
        "keep_labels": keep_labels.float(),
        "local_accept_labels": local_accept.float(),
        "parent_indices": parent_indices,
        "target_token_labels": target_argmax,
        "node_mask": valid_nodes,
    }


def draft_verifier_loss(
    token_logits: Tensor,
    target_logits: Tensor,
    draft_token_ids: Tensor,
    tree_mask: Tensor,
    tree_position_ids: Tensor,
    keep_labels: Tensor | None = None,
    node_scores: Tensor | None = None,
    node_mask: Tensor | None = None,
    token_labels: Tensor | None = None,
    parent_indices: Tensor | None = None,
    distill_weight: float = 1.0,
    token_weight: float = 0.5,
    edge_weight: float = 1.0,
    branch_weight: float = 0.0,
    temperature: float = 1.0,
    offpolicy_distill_weight: float = 0.1,
    offpolicy_token_weight: float | None = None,
    rejected_edge_weight: float = 1.0,
    dead_edge_weight: float = 0.25,
    ignore_index: int = -100,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """
    Loss for the frozen-draft, one-pass verifier stage.

    Main-context KD/CE uses accepted nodes, including the parent of the first
    rejected token. Rejected/off-policy node contexts can still receive a small
    KD/CE weight so the verifier remains target-like without overfitting dead
    continuations. Edge loss scores each draft child from its parent's verifier
    logits and supplies the explicit negative signal for rejected draft tokens.
    """
    if token_logits.dim() != 3 or target_logits.dim() != 3:
        raise ValueError("token_logits and target_logits must be [batch, nodes, vocab]")
    if token_logits.shape[:2] != target_logits.shape[:2]:
        raise ValueError("token_logits and target_logits must share batch/node shape")
    if token_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError(
            "verifier and target logits must be in the same vocab space before KD"
        )

    batch_size, node_count, _ = token_logits.shape
    device = token_logits.device
    target_logits = target_logits.to(device).detach()
    draft_token_ids = _ensure_batch_nodes(draft_token_ids, batch_size, node_count).to(device)
    valid_nodes = _valid_node_mask(node_mask, batch_size, node_count, device)
    if parent_indices is None:
        parent_indices = tree_parent_indices(
            tree_mask=tree_mask,
            position_ids=tree_position_ids,
            batch_size=batch_size,
        ).to(device)
    else:
        parent_indices = _ensure_batch_nodes(parent_indices, batch_size, node_count).to(device)

    if keep_labels is None:
        label_info = build_accept_labels_from_target_logits(
            target_logits=target_logits,
            draft_token_ids=draft_token_ids,
            tree_mask=tree_mask,
            tree_position_ids=tree_position_ids,
            node_mask=valid_nodes,
            parent_indices=parent_indices,
        )
        keep = label_info["keep_labels"].to(device).bool()
    else:
        keep = _ensure_batch_nodes(keep_labels, batch_size, node_count).to(device).bool()
        keep = keep & valid_nodes
        keep[:, 0] = valid_nodes[:, 0]

    if token_labels is None:
        token_labels = target_logits.argmax(dim=-1)
    token_labels = _ensure_batch_nodes(token_labels, batch_size, node_count).to(device)

    if offpolicy_token_weight is None:
        offpolicy_token_weight = offpolicy_distill_weight
    context_weights = torch.where(
        keep,
        torch.ones((), device=device, dtype=token_logits.dtype),
        torch.full((), offpolicy_distill_weight, device=device, dtype=token_logits.dtype),
    )
    context_weights = context_weights * valid_nodes.to(token_logits.dtype)
    token_context_weights = torch.where(
        keep,
        torch.ones((), device=device, dtype=token_logits.dtype),
        torch.full((), offpolicy_token_weight, device=device, dtype=token_logits.dtype),
    )
    token_context_weights = token_context_weights * valid_nodes.to(token_logits.dtype)
    token_context_weights = token_context_weights * token_labels.ne(ignore_index).to(
        token_logits.dtype
    )

    student_logp = F.log_softmax(token_logits.float() / temperature, dim=-1)
    teacher_p = F.softmax(target_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(target_logits.float() / temperature, dim=-1)
    per_node_kl = torch.sum(
        teacher_p * (teacher_logp - student_logp),
        dim=-1,
    ) * temperature**2
    distill_loss = _weighted_mean(per_node_kl, context_weights, eps=eps)

    per_node_token = F.cross_entropy(
        token_logits.flatten(0, 1).float(),
        token_labels.flatten(),
        reduction="none",
        ignore_index=ignore_index,
    ).view(batch_size, node_count)
    token_loss = _weighted_mean(per_node_token, token_context_weights, eps=eps)

    edge_loss = token_logits.new_zeros(())
    if node_count > 1:
        child_ids = torch.arange(node_count, device=device)[None].expand(batch_size, -1)
        edge_mask = valid_nodes & child_ids.ne(0) & parent_indices.ge(0)
        safe_parent = parent_indices.clamp_min(0)
        batch_ids = torch.arange(batch_size, device=device)[:, None]
        parent_logp = F.log_softmax(token_logits.float(), dim=-1)[batch_ids, safe_parent]
        child_logp = parent_logp.gather(
            dim=-1,
            index=draft_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        child_prob = child_logp.exp().clamp(max=1.0 - eps)
        accepted_edge = keep & edge_mask
        rejected_edge = (~keep) & edge_mask
        parent_keep = keep[batch_ids, safe_parent]
        first_reject_edge = rejected_edge & parent_keep
        dead_edge = rejected_edge & (~parent_keep)
        edge_weights = torch.zeros_like(child_prob)
        edge_weights = torch.where(
            accepted_edge,
            torch.ones_like(edge_weights),
            edge_weights,
        )
        edge_weights = torch.where(
            first_reject_edge,
            torch.full_like(edge_weights, rejected_edge_weight),
            edge_weights,
        )
        edge_weights = torch.where(
            dead_edge,
            torch.full_like(edge_weights, dead_edge_weight),
            edge_weights,
        )
        per_edge_loss = torch.where(
            accepted_edge,
            -child_logp,
            -torch.log1p(-child_prob),
        )
        edge_loss = _weighted_mean(per_edge_loss, edge_weights, eps=eps)

    branch_loss = token_logits.new_zeros(())
    if node_scores is not None and branch_weight:
        score = _ensure_batch_nodes(node_scores, batch_size, node_count).to(device)
        branch_weights = valid_nodes.to(score.dtype)
        branch_weights[:, 0] = 0
        per_node_branch = F.binary_cross_entropy_with_logits(
            score.float(),
            keep.float(),
            reduction="none",
        )
        branch_loss = _weighted_mean(per_node_branch, branch_weights, eps=eps)

    loss = (
        distill_weight * distill_loss
        + token_weight * token_loss
        + edge_weight * edge_loss
        + branch_weight * branch_loss
    )
    return {
        "loss": loss,
        "distill": distill_loss.detach(),
        "token": token_loss.detach(),
        "edge": edge_loss.detach(),
        "branch": branch_loss.detach(),
        "accepted_nodes": (keep & valid_nodes).sum().detach(),
        "valid_nodes": valid_nodes.sum().detach(),
    }


def branch_refiner_loss(
    node_scores: Tensor,
    token_logits: Tensor,
    keep_labels: Tensor,
    token_labels: Tensor | None = None,
    teacher_logits: Tensor | None = None,
    keep_weight: float = 1.0,
    token_weight: float = 1.0,
    distill_weight: float = 0.0,
    temperature: float = 1.0,
    ignore_index: int = -100,
    keep_mask: Tensor | None = None,
    token_mask: Tensor | None = None,
    distill_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    device = node_scores.device
    keep_labels = keep_labels.to(device).float()
    keep_weights = (
        torch.ones_like(keep_labels, dtype=node_scores.dtype)
        if keep_mask is None
        else keep_mask.to(device).to(node_scores.dtype)
    )
    per_keep = F.binary_cross_entropy_with_logits(
        node_scores.float(),
        keep_labels,
        reduction="none",
    )
    keep_loss = _weighted_mean(per_keep, keep_weights)

    token_loss = node_scores.new_zeros(())
    if token_labels is not None:
        token_labels = token_labels.to(token_logits.device)
        per_token = F.cross_entropy(
            token_logits.flatten(0, 1).float(),
            token_labels.flatten(),
            reduction="none",
            ignore_index=ignore_index,
        ).view(token_logits.shape[:2])
        token_weights = token_labels.ne(ignore_index).to(token_logits.dtype)
        if token_mask is not None:
            token_weights = token_weights * token_mask.to(token_logits.device).to(
                token_logits.dtype
            )
        token_loss = _weighted_mean(per_token, token_weights)

    distill_loss = node_scores.new_zeros(())
    if teacher_logits is not None:
        student = F.log_softmax(token_logits.float() / temperature, dim=-1)
        teacher_logits = teacher_logits.to(token_logits.device).float().detach()
        teacher = F.softmax(teacher_logits / temperature, dim=-1)
        teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
        per_distill = torch.sum(teacher * (teacher_logp - student), dim=-1) * temperature**2
        distill_weights = torch.ones_like(per_distill, dtype=token_logits.dtype)
        if distill_mask is not None:
            distill_weights = distill_mask.to(token_logits.device).to(token_logits.dtype)
        distill_loss = _weighted_mean(per_distill, distill_weights)

    loss = (
        keep_weight * keep_loss
        + token_weight * token_loss
        + distill_weight * distill_loss
    )
    return {
        "loss": loss,
        "keep": keep_loss.detach(),
        "token": token_loss.detach(),
        "distill": distill_loss.detach(),
    }


def _ensure_batch_nodes(tensor: Tensor, batch_size: int, node_count: int) -> Tensor:
    if tensor.dim() == 1:
        tensor = tensor[None]
    if tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1)
    if tensor.shape[:2] != (batch_size, node_count):
        raise ValueError(
            f"expected shape [{batch_size}, {node_count}], got {tuple(tensor.shape)}"
        )
    return tensor


def _valid_node_mask(
    node_mask: Tensor | None,
    batch_size: int,
    node_count: int,
    device: torch.device,
) -> Tensor:
    if node_mask is None:
        return torch.ones(batch_size, node_count, dtype=torch.bool, device=device)
    return _ensure_batch_nodes(node_mask, batch_size, node_count).to(device).bool()


def _weighted_mean(values: Tensor, weights: Tensor, eps: float = 1e-6) -> Tensor:
    weights = weights.to(values.device).to(values.dtype)
    denom = weights.sum().clamp_min(eps)
    return torch.sum(values * weights) / denom
