# Draft Verifier For EAGLE-3

This clone adds an opt-in second draft stage on top of the official EAGLE-3 tree generator.

## Intuition

The idea is reasonable, with one important constraint: the second draft verifier should not be treated as a lossless verifier. It is a learned reranker/pruner. The target model still performs the final speculative verification, so generation remains exact under the existing EAGLE acceptance rule.

In this implementation:

- EAGLE-3 first builds its usual wide draft tree.
- `DraftBranchRefiner` runs once over the proposed tree using an EAGLE-style decoder block.
- It reuses the draft model's own token embeddings and LM head.
- It consumes the per-node draft hidden state that produced each draft token.
- It uses the tree attention mask plus RoPE `position_ids` for branch/depth structure instead of adding separate learned depth embeddings.
- It either scores existing branches, or rescores the proposed draft tokens with the full shared LM head.
- It keeps a configurable number of children per depth, such as `[1, 1, 1, 2, 2, 2, 2]`.
- The target model verifies only the refined tree.

This matches the "two-layer EAGLE" picture: layer one proposes many branches autoregressively, layer two refines the whole proposed tree in one parallel pass.

## Which Mode Is Better

Use `branch_refiner_mode: "rescore"` for the main experiment. It is closer to your two-layer EAGLE idea because the second draft layer produces full draft-vocab logits through the same LM head and can change the ranking of draft tokens. A child branch is scored from its parent's verifier logits, matching EAGLE's autoregressive contract.

Keep `branch_refiner_mode: "score"` as the cheap ablation. It is faster and easier to train, but it can only learn "is this branch worth target verification?" rather than "among these sibling tokens, which one should move up?"

## Where The Code Is

- `eagle/model/branch_refiner.py`: one-pass branch refiner, tree pruning, and starter loss helpers.
- `eagle/model/cnets.py`: guarded integration in `Model.topK_genrate`.
- `eagle/model/configs.py` and `eagle/traineagle3/configs.py`: config flags.
- `eagle/traineagle3/config_draft_verifier.json`: example config with the refiner enabled.

## Config

The official path stays unchanged unless `use_branch_refiner` is true.

```json
{
  "draft_topk_schedule": [8, 8, 8, 8, 8, 8, 8],
  "use_branch_refiner": true,
  "branch_refiner_mode": "rescore",
  "draft_depth_range": [1, 8],
  "branch_refiner_keep_schedule": [1, 1, 1, 2, 2, 2, 2],
  "branch_refiner_keep_per_depth": 2,
  "branch_refiner_max_depth": 8,
  "branch_refiner_max_tokens": 32,
  "branch_refiner_num_hidden_layers": 1,
  "branch_refiner_draft_score_weight": 1.0,
  "branch_refiner_token_score_weight": 1.0,
  "branch_refiner_branch_score_weight": 1.0,
  "branch_refiner_distill_weight": 1.0,
  "branch_refiner_token_ce_weight": 0.5,
  "branch_refiner_edge_weight": 1.0,
  "branch_refiner_branch_bce_weight": 0.0,
  "branch_refiner_temperature": 1.0,
  "branch_refiner_offpolicy_distill_weight": 0.1,
  "branch_refiner_offpolicy_token_weight": 0.1,
  "branch_refiner_rejected_edge_weight": 1.0,
  "branch_refiner_dead_edge_weight": 0.25
}
```

`draft_topk_schedule` controls the first autoregressive EAGLE draft stage. If it is omitted, official scalar `top_k` behavior is used. `branch_refiner_keep_schedule` controls how many children the verifier draft keeps per depth after scoring/rescoring. If the schedule is shorter than the generated depth, the last value is reused.

`draft_depth_range` makes the draft tree depth variable using the same internal meaning as EAGLE's `depth` argument. `[1, 8]` samples a new generation depth each call, so the verifier does not become a fixed-depth model. Leave it unset for the original fixed `depth` behavior.

`branch_refiner_keep_per_depth` is the scalar fallback. `branch_refiner_max_tokens` is the cap on non-root draft nodes sent to the target verifier after pruning.

## Training Plan

1. Train the normal EAGLE-3 draft model with the official EAGLE method.
2. Freeze the target model and freeze the first EAGLE draft. The verifier is not used while the first draft is being trained. In code, call `model.ea_layer.freeze_draft_for_verifier_training()` after loading the trained draft with `use_branch_refiner` enabled.
3. For each prompt, sample a draft depth from `draft_depth_range`, then let the frozen draft generate the tree and save the draft hidden state that produced each node.
4. Run the target model on the same draft tree in teacher-forced/tree-verification mode. This is allowed even when the draft tree contains wrong tokens because the target is scoring the context, not generating from it.
5. Train `DraftBranchRefiner.forward` with `draft_verifier_loss` from `eagle/model/branch_refiner.py`.

The verifier loss is:

```text
loss =
  distill_weight * KD(verifier_logits, target_logits)
  + token_ce_weight * CE(verifier_logits, target_argmax)
  + edge_weight * edge_accept_loss(parent_logits, child_token, keep_label)
  + optional branch_bce_weight * BCE(node_score, keep_label)
```

`KD` and `CE` use full weight on accepted contexts, including the parent context where the first rejection happens. Rejected/off-policy contexts after a failed token get a small weight through `branch_refiner_offpolicy_distill_weight`, default `0.1`, so the verifier stays target-like without spending most of its capacity on dead continuations.

`edge_accept_loss` scores every proposed child from its parent's verifier logits:

```text
accepted child: -log P_verifier(child_token | parent)
rejected child: -log(1 - P_verifier(child_token | parent))
```

For a path `A -> C -> K -> M` where target accepts `A`, accepts `C`, and rejects `K`, the labels are:

```text
A = 1
C = 1
K = 0
M = 0
```

The high-weight KD/CE at context `A,C` teaches the verifier to predict target token `H` instead of draft token `K`. The explicit edge penalty teaches it not to send `K` to the target. The continuation after `K` can still receive low-weight KD if enabled, but it is not treated as a live acceptance path.

6. Tune inference with `total_token=60`, first-stage `top_k=8-10`, `branch_refiner_keep_per_depth=2`, and `branch_refiner_max_tokens` in the 24-40 range.

The first useful ablation is acceptance tokens per target forward versus target tree size. The second is end-to-end tokens/sec, because a better refiner can still lose if it adds too much latency.

## Ground Truth

For greedy/speculative decoding, the clean ground truth is the target model's own verification result. For each edge `parent -> child`, use the target logits at the parent context. If `child_token == argmax(target_logits[parent])`, that edge is locally correct. A node is path-positive only if all previous edges are also locally correct.

The refiner's full LM-head logits are trained on target logits in the same draft-vocab space. If `draft_vocab_size < vocab_size`, restrict/map target logits to the draft vocabulary before calling `draft_verifier_loss`.

At inference, the child score combines:

- first EAGLE draft path score,
- verifier branch utility score,
- verifier log-prob of the child token from the parent node.

## Main Risk

If the correct token is often outside the first draft top-k, the second refiner cannot recover it because this version reranks existing nodes rather than proposing new tokens. A later version can add a small `lm_head` expansion step so the refiner can replace bad children, not only prune them.
