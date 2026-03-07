# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    One advantage computation modes:

    1. "turn": Turn-level GRPO
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    # Handle rewards shape - squeeze if needed
    if rewards.ndim == 2:
        rewards_flat = rewards.squeeze(-1)  # [num_sequence, 1] -> [num_sequence]
    else:
        rewards_flat = rewards  # Already [num_sequence]

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    # Determine number of questions
    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    # Initialize advantage tensor
    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )
    if advantage_mode == "turn":
        # For each question, compute GRPO over all its turns

        # Step 1: Map each turn to its question
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: Compute per-question statistics over all turns
        for question_idx in range(num_questions):
            # Get all turns belonging to this question
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Compute statistics for this question's turns
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Normalize turns in this question
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Assign back to turn_advantages
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(f"Invalid advantage_mode: {advantage_mode}. Must be 'turn'")

    # Broadcast advantages to match loss_mask shape [seq_len, num_sequence]
    # turn_advantages is [num_sequence], we broadcast to [seq_len, num_sequence]
    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("regrpo")
def compute_regrpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    t_clip: torch.Tensor,
    p_flip: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    **kwargs,
):
    """
    Compute ReGRPO advantages with flip rate from rewriting.

    ReGRPO redistributes advantages based on prefix-suffix contribution analysis.
    The weight is determined by the flip rate (p_flip) computed from rewriting
    trajectories from the lowest logprob position (t_clip).

    Weight formula:
    - prefix (t < t_clip): w_t = (1 - p_flip) / t_clip
    - suffix (t >= t_clip): w_t = p_flip / (T - t_clip)

    Args:
        rewards: Reward scores after calculate_scores. Shape: [num_groups, group_size]
        loss_mask: Loss mask. Shape: [n_steps, batch_size]
        group_size: Group size for GRPO advantage computation.
        t_clip: Rewrite position for each trajectory. Shape: [batch_size]
        p_flip: Flip rate for each trajectory. Shape: [batch_size]
        num_chunk: Number of chunks (T).
        chunk_size: Size of each chunk (for expanding to step level).

    Returns:
        advantages: Shape [n_steps, batch_size]
        returns: None
    """
    T = num_chunk
    batch_size = t_clip.shape[0]
    device = rewards.device

    # Ensure t_clip and p_flip are on the same device as rewards
    t_clip = t_clip.to(device)
    p_flip = p_flip.to(device)

    # 1. Compute original GRPO advantage (per trajectory)
    grouped_rewards = rewards.view(-1, group_size)
    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True)
    A_traj = (grouped_rewards - grouped_reward_mean) / (grouped_reward_std + 1e-6)
    A_traj_flat = A_traj.view(-1)  # [batch_size]

    # 2. Compute weight distribution based on t_clip and p_flip
    #
    # CRITICAL: use group-level shared t_clip and p_flip for all trajectories in a group.
    # Each trajectory has its own t_clip/p_flip, causing different chunk_weight distributions.
    # Multiplying A_traj (zero-sum at trajectory level) by different weight distributions
    # yields chunk_advantages that are NOT zero-sum → systematic drift of t_clip and collapse.
    # Sharing group-mean t_clip/p_flip ensures identical weight functions within a group,
    # preserving chunk-level zero-sum at every position.
    t_clip_shared = (
        t_clip.float().view(-1, group_size).mean(dim=-1, keepdim=True)
        .round().long()
        .expand(-1, group_size).reshape(-1)
    )
    p_flip_shared = (
        p_flip.float().view(-1, group_size).mean(dim=-1, keepdim=True)
        .expand(-1, group_size).reshape(-1)
    )

    # chunk_indices: [T, 1]
    chunk_indices = torch.arange(T, device=device).unsqueeze(1).float()
    # t_clip_expanded: [1, batch_size]
    t_clip_expanded = t_clip_shared.unsqueeze(0).float()

    # Create masks for prefix and suffix
    prefix_mask = chunk_indices < t_clip_expanded  # [T, batch_size]
    suffix_mask = ~prefix_mask

    # p_flip_expanded: [1, batch_size]
    p_flip_expanded = p_flip_shared.unsqueeze(0).float()

    # Avoid division by zero
    t_clip_safe = t_clip_expanded.clamp(min=1)
    suffix_len_safe = (T - t_clip_expanded).clamp(min=1)

    # Compute weights per chunk
    prefix_weight = (1 - p_flip_expanded) / t_clip_safe
    suffix_weight = p_flip_expanded / suffix_len_safe

    # Combine weights: [T, batch_size]
    chunk_weights = prefix_mask.float() * prefix_weight + suffix_mask.float() * suffix_weight

    # Normalize weights so they sum to 1 for each trajectory
    weight_sum = chunk_weights.sum(dim=0, keepdim=True)
    chunk_weights = chunk_weights / (weight_sum + 1e-8)

    # 3. Apply weights to advantage
    # A_traj_flat: [batch_size] -> [1, batch_size]
    chunk_advantages = A_traj_flat.unsqueeze(0) * chunk_weights * T  # [T, batch_size]

    # Cap amplification: bound per-chunk weight at max_amp/T to prevent excessive
    # credit concentration on a small prefix (e.g. t_clip=5 → 6.4× amplification).
    max_amp = kwargs.get("regrpo_max_amp", 3.0)  # max amplification vs vanilla GRPO
    weight_cap = max_amp / T
    chunk_weights_capped = chunk_weights.clamp(max=weight_cap)

    # Modulation coefficient α ∈ [0, 1]:
    #   α=0 → vanilla GRPO (uniform weight 1/T for all chunks, no redistribution)
    #   α=1 → full ReGRPO redistribution (prefix/suffix weighted by t_clip/p_flip)
    #   0<α<1 → interpolation: w_mixed = (1-α)×(1/T) + α×w_capped
    # This ensures chunk_advantages remain zero-sum at every chunk position regardless of α,
    # while smoothly controlling how aggressively the temporal credit is redistributed.
    alpha = kwargs.get("regrpo_alpha", 1.0)
    w_vanilla = torch.ones(T, batch_size, device=device) / T  # uniform: [T, batch_size]
    chunk_weights_mixed = (1.0 - alpha) * w_vanilla + alpha * chunk_weights_capped
    chunk_advantages = A_traj_flat.unsqueeze(0) * chunk_weights_mixed * T  # [T, batch_size]

    # Debug: save key intermediate variables to txt
    import os
    rank = int(os.environ.get("RANK", 0))
    debug_dir = "./tmp/regrpo_debug-beta0"
    os.makedirs(debug_dir, exist_ok=True)
    existing = len([f for f in os.listdir(debug_dir) if f.startswith(f"rank{rank:02d}_step_")])
    debug_path = os.path.join(debug_dir, f"rank{rank:02d}_step_{existing:04d}.txt")
    with open(debug_path, "w") as f:
        f.write(f"===== ReGRPO Debug rank={rank} (step {existing}) =====\n\n")
        f.write(f"group_size={group_size}, num_chunk(T)={T}, chunk_size={chunk_size}\n")
        f.write(f"batch_size={batch_size}\n\n")

        f.write("--- grouped_rewards (per group, shape [num_groups, group_size]) ---\n")
        f.write(f"{grouped_rewards}\n\n")

        f.write("--- grouped_reward_mean ---\n")
        f.write(f"{grouped_reward_mean.squeeze()}\n\n")

        f.write("--- grouped_reward_std ---\n")
        f.write(f"{grouped_reward_std.squeeze()}\n\n")

        f.write("--- A_traj (normalized advantage per traj, shape [num_groups, group_size]) ---\n")
        f.write(f"{A_traj}\n\n")

        f.write("--- A_traj_flat (shape [batch_size]) ---\n")
        f.write(f"{A_traj_flat}\n\n")

        f.write("--- t_clip (individual, shape [batch_size]) ---\n")
        f.write(f"{t_clip}\n\n")

        f.write("--- t_clip_shared (group-mean, shape [batch_size]) ---\n")
        f.write(f"{t_clip_shared}\n\n")

        f.write("--- p_flip (individual, shape [batch_size]) ---\n")
        f.write(f"{p_flip}\n\n")

        f.write("--- p_flip_shared (group-mean, shape [batch_size]) ---\n")
        f.write(f"{p_flip_shared}\n\n")

        f.write("--- chunk_weights (shape [T, batch_size]) ---\n")
        f.write(f"{chunk_weights}\n\n")

        f.write("--- chunk_advantages (shape [T, batch_size]) ---\n")
        f.write(f"{chunk_advantages}\n\n")
    print(f"[ReGRPO Debug] saved to {debug_path}")
    # 4. Expand from chunk level to action/step level
    # Repeat each chunk value for chunk_size steps
    advantages = chunk_advantages.repeat_interleave(chunk_size, dim=0)  # [n_steps, batch_size]

    # Apply loss mask
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("segrpo")
def compute_segrpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    t_clip: torch.Tensor,
    rewrite_rewards: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    **kwargs,
):
    """
    Compute SeGRPO advantages: GRPO-style normalization on rewrite rewards,
    applied only to suffix chunks (t >= t_clip).

    The original rollout is used only to find t_clip.  Rewrite trajectories
    (started from t_clip) provide the reward signal.  Advantages are zeroed
    out for prefix chunks so gradients flow only through the suffix.

    Args:
        rewards: Original trajectory rewards (unused for A computation, kept
            for API consistency). Shape: [num_groups, group_size]
        loss_mask: Loss mask. Shape: [n_steps, batch_size]
        group_size: Group size for GRPO normalization.
        t_clip: Rewrite start chunk for each trajectory. Shape: [batch_size]
        rewrite_rewards: Mean reward over rewrites per trajectory.
            Shape: [batch_size]
        num_chunk: Number of chunks (T).
        chunk_size: Steps per chunk.

    Returns:
        advantages: Shape [n_steps, batch_size]
        returns: None
    """
    T = num_chunk
    batch_size = t_clip.shape[0]
    device = rewards.device

    t_clip = t_clip.to(device)
    rewrite_rewards = rewrite_rewards.to(device)

    # 1. GRPO normalization on rewrite rewards within each group
    grouped_rw = rewrite_rewards.view(-1, group_size)
    grouped_rw_mean = grouped_rw.mean(dim=-1, keepdim=True)
    grouped_rw_std = grouped_rw.std(dim=-1, keepdim=True)
    A_traj = (grouped_rw - grouped_rw_mean) / (grouped_rw_std + 1e-6)
    A_traj_flat = A_traj.view(-1)  # [batch_size]

    # 2. Build suffix mask: 1 for t >= t_clip, 0 for prefix
    chunk_indices = torch.arange(T, device=device).unsqueeze(1).float()  # [T, 1]
    t_clip_expanded = t_clip.unsqueeze(0).float()  # [1, batch_size]
    suffix_mask = (chunk_indices >= t_clip_expanded).float()  # [T, batch_size]

    # 3. Assign A_traj to suffix chunks scaled to match GRPO total gradient magnitude
    suffix_len = (T - t_clip).clamp(min=1).float()  # [batch_size]
    # Scale by T/suffix_len so the total gradient contribution per traj equals T*A_traj,
    # same as vanilla GRPO. This concentrates the signal in the suffix while keeping
    # the same learning rate as GRPO. Clamp prevents explosion when suffix is very short.
    chunk_advantages = A_traj_flat.unsqueeze(0) * suffix_mask * (T / suffix_len.unsqueeze(0))  # [T, batch_size]
    chunk_advantages = chunk_advantages.clamp(-5.0, 5.0)

    # Alpha modulation: α=0 → no segrpo signal, α=1 → full segrpo
    alpha = kwargs.get("regrpo_alpha", 1.0)
    chunk_advantages = chunk_advantages * alpha

    # Debug: save key intermediate variables to txt
    import os
    rank = int(os.environ.get("RANK", 0))
    debug_dir = "./tmp/segrpo_debug"
    os.makedirs(debug_dir, exist_ok=True)
    existing = len([f for f in os.listdir(debug_dir) if f.startswith(f"rank{rank:02d}_step_")])
    debug_path = os.path.join(debug_dir, f"rank{rank:02d}_step_{existing:04d}.txt")
    with open(debug_path, "w") as f:
        f.write(f"===== SeGRPO Debug rank={rank} (step {existing}) =====\n\n")
        f.write(f"group_size={group_size}, num_chunk(T)={T}, chunk_size={chunk_size}\n")
        f.write(f"batch_size={batch_size}\n\n")

        f.write("--- rewrite_rewards (mean over rewrites per traj, shape [batch_size]) ---\n")
        f.write(f"{rewrite_rewards}\n\n")

        f.write("--- grouped_rw (shape [num_groups, group_size]) ---\n")
        f.write(f"{grouped_rw}\n\n")

        f.write("--- grouped_rw_mean ---\n")
        f.write(f"{grouped_rw_mean.squeeze()}\n\n")

        f.write("--- grouped_rw_std ---\n")
        f.write(f"{grouped_rw_std.squeeze()}\n\n")

        f.write("--- A_traj (normalized advantage, shape [num_groups, group_size]) ---\n")
        f.write(f"{A_traj}\n\n")

        f.write("--- A_traj_flat (shape [batch_size]) ---\n")
        f.write(f"{A_traj_flat}\n\n")

        f.write("--- t_clip (shape [batch_size]) ---\n")
        f.write(f"{t_clip}\n\n")

        f.write("--- suffix_mask (shape [T, batch_size]) ---\n")
        f.write(f"{suffix_mask}\n\n")

        f.write("--- chunk_advantages (shape [T, batch_size]) ---\n")
        f.write(f"{chunk_advantages}\n\n")

        f.write("--- chunk_advantages stats ---\n")
        # Verify prefix is zeroed: for each traj, chunks t < t_clip should be 0
        prefix_leak = 0.0
        for traj_i in range(batch_size):
            tc = int(t_clip[traj_i].item())
            if tc > 0:
                prefix_leak += chunk_advantages[:tc, traj_i].abs().sum().item()
        f.write(f"  prefix zero-leak (should be 0): {prefix_leak:.6f}\n")
        f.write(f"  nonzero chunks: {(chunk_advantages.abs() > 1e-6).sum().item()}\n")
        f.write(f"  max={chunk_advantages.max().item():.4f}, min={chunk_advantages.min().item():.4f}\n")
        f.write(f"  mean(nonzero)={chunk_advantages[chunk_advantages.abs() > 1e-6].mean().item():.4f}\n")
        # Per-traj summary
        f.write("\n--- per-traj summary (t_clip | suffix_len | A_traj | rewrite_reward) ---\n")
        for traj_i in range(min(batch_size, 16)):
            tc = int(t_clip[traj_i].item())
            sl = T - tc
            a = A_traj_flat[traj_i].item()
            rw = rewrite_rewards[traj_i].item()
            f.write(f"  traj{traj_i:02d}: t_clip={tc:2d}, suffix_len={sl:2d}, A_traj={a:+.4f}, rewrite_reward={rw:.4f}\n")
    print(f"[SeGRPO Debug] saved to {debug_path}")

    # 4. Expand to step level and apply loss mask
    advantages = chunk_advantages.repeat_interleave(chunk_size, dim=0)  # [n_steps, batch_size]
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("fullgrpo")
def compute_fullgrpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    t_clip: torch.Tensor,
    p_flip: torch.Tensor,
    rewrite_rewards: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    **kwargs,
):
    """
    Compute FullGRPO advantages: ReGRPO on original rollout + SeGRPO on rewrite rollout.

    Two gradient sources are combined:
    - ReGRPO part: original trajectory rewards, redistributed to prefix/suffix by
      (t_clip, p_flip), applied to ALL chunks.
    - SeGRPO part: rewrite trajectory rewards, GRPO-normalized within each group,
      applied only to suffix chunks (t >= t_clip).

    The final per-chunk advantage is the element-wise sum of both parts.

    Args:
        rewards: Original trajectory rewards. Shape: [num_groups, group_size]
        loss_mask: Loss mask. Shape: [n_steps, batch_size]
        group_size: Group size for GRPO normalization.
        t_clip: Rewrite start chunk per trajectory. Shape: [batch_size]
        p_flip: Flip rate per trajectory. Shape: [batch_size]
        rewrite_rewards: Mean rewrite reward per trajectory. Shape: [batch_size]
        num_chunk: Number of chunks (T).
        chunk_size: Steps per chunk.

    Returns:
        advantages: Shape [n_steps, batch_size]
        returns: None
    """
    T = num_chunk
    batch_size = t_clip.shape[0]
    device = rewards.device

    t_clip = t_clip.to(device)
    p_flip = p_flip.to(device)
    rewrite_rewards = rewrite_rewards.to(device)

    # ── Part 1: ReGRPO on original rollout ──────────────────────────────────
    grouped_rewards = rewards.view(-1, group_size)
    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True)
    A_traj = (grouped_rewards - grouped_reward_mean) / (grouped_reward_std + 1e-6)
    A_traj_flat = A_traj.view(-1)  # [batch_size]

    chunk_indices = torch.arange(T, device=device).unsqueeze(1).float()
    t_clip_expanded = t_clip.unsqueeze(0).float()
    p_flip_expanded = p_flip.unsqueeze(0).float()

    prefix_mask = chunk_indices < t_clip_expanded
    suffix_mask_bool = ~prefix_mask
    t_clip_safe = t_clip_expanded.clamp(min=1)
    suffix_len_safe = (T - t_clip_expanded).clamp(min=1)

    prefix_weight = (1 - p_flip_expanded) / t_clip_safe
    suffix_weight = p_flip_expanded / suffix_len_safe
    chunk_weights = prefix_mask.float() * prefix_weight + suffix_mask_bool.float() * suffix_weight
    weight_sum = chunk_weights.sum(dim=0, keepdim=True)
    chunk_weights = chunk_weights / (weight_sum + 1e-8)

    regrpo_chunk_adv = A_traj_flat.unsqueeze(0) * chunk_weights * T  # [T, batch_size]
    regrpo_chunk_adv = regrpo_chunk_adv.clamp(-5.0, 5.0)

    # ── Part 2: SeGRPO on rewrite rollout (suffix only) ─────────────────────
    grouped_rw = rewrite_rewards.view(-1, group_size)
    grouped_rw_mean = grouped_rw.mean(dim=-1, keepdim=True)
    grouped_rw_std = grouped_rw.std(dim=-1, keepdim=True)
    A_rw = (grouped_rw - grouped_rw_mean) / (grouped_rw_std + 1e-6)
    A_rw_flat = A_rw.view(-1)  # [batch_size]

    suffix_mask_float = suffix_mask_bool.float()
    segrpo_chunk_adv = A_rw_flat.unsqueeze(0) * suffix_mask_float  # [T, batch_size]

    # ── Combine ──────────────────────────────────────────────────────────────
    chunk_advantages = regrpo_chunk_adv + segrpo_chunk_adv  # [T, batch_size]

    # 4. Expand to step level and apply loss mask
    advantages = chunk_advantages.repeat_interleave(chunk_size, dim=0)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None
