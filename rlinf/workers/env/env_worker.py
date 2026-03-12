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

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )

        # ReGRPO/SeGRPO/FullGRPO: enable chunk state saving for rewriting
        self.enable_regrpo = cfg.algorithm.get("adv_type", "") in ("regrpo", "segrpo", "fullgrpo")
        self.regrpo_num_rewrite = cfg.algorithm.get("regrpo", {}).get("num_rewrite", 2)
        self.regrpo_success_threshold = cfg.algorithm.get("regrpo", {}).get(
            "success_threshold", 0.5
        )
        self.regrpo_min_prefix_chunks = cfg.algorithm.get("regrpo", {}).get(
            "min_prefix_chunks", 1
        )
        self.regrpo_max_prefix_chunks = cfg.algorithm.get("regrpo", {}).get(
            "max_prefix_chunks", None
        )
        # Storage for chunk states, actions, and returns (per stage, per epoch)
        # chunk_states_buffer[stage_id][epoch][chunk_idx] = bytes
        self.chunk_states_buffer: list[list[list[bytes]]] = []
        # chunk_actions_buffer[stage_id][epoch][chunk_idx] = tensor
        self.chunk_actions_buffer: list[list[list[torch.Tensor]]] = []
        # chunk_rewards_buffer[stage_id][epoch][chunk_idx] = tensor [num_envs]
        self.chunk_rewards_buffer: list[list[list[torch.Tensor]]] = []
        # chunk_logprobs_buffer[stage_id][epoch][chunk_idx] = tensor [num_envs, ...]
        self.chunk_logprobs_buffer: list[list[list[torch.Tensor]]] = []
        # chunk_termination_idx_buffer[stage_id][epoch] = tensor [num_envs] recording first termination chunk index
        # If no termination, value is n_chunk_steps (meaning all chunks are valid)
        self.chunk_termination_idx_buffer: list[list[torch.Tensor]] = []
        # epoch_returns_buffer[stage_id][epoch] = tensor of returns per env
        self.epoch_returns_buffer: list[list[torch.Tensor]] = []
        # Final p_flip, t_clip, and rewrite_rewards for all trajectories
        self.regrpo_p_flip: torch.Tensor = None
        self.regrpo_t_clip: torch.Tensor = None
        self.segrpo_rewrite_rewards: torch.Tensor = None

    def init_worker(self):
        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }

        # ReGRPO: reuse train ranks for rewriting
        if self.enable_regrpo:
            self.dst_ranks["rewrite"] = self.dst_ranks["train"]
            self.src_ranks["rewrite"] = self.src_ranks["train"]

        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
        self.log_info(f"Env worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Env worker initialized with src_ranks: {self.src_ranks}")
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                env = train_env_cls(
                    cfg=self.cfg.env.train,
                    num_envs=self.train_num_envs_per_stage,
                    seed_offset=self._rank * self.stage_num + stage_id,
                    total_num_processes=self._world_size * self.stage_num,
                    worker_info=self.worker_info,
                )
                if self.cfg.env.train.video_cfg.save_video:
                    env = RecordVideo(env, self.cfg.env.train.video_cfg)
                self.env_list.append(env)
        if self.enable_eval:
            for stage_id in range(self.stage_num):
                env = eval_env_cls(
                    cfg=self.cfg.env.eval,
                    num_envs=self.eval_num_envs_per_stage,
                    seed_offset=self._rank * self.stage_num + stage_id,
                    total_num_processes=self._world_size * self.stage_num,
                    worker_info=self.worker_info,
                )
                if self.cfg.env.eval.video_cfg.save_video:
                    env = RecordVideo(env, self.cfg.env.eval.video_cfg)
                self.eval_env_list.append(env)

        if not self.only_eval:
            self._init_env()

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout peer ranks for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(rollout_rank, batch_size)`` tuples this env worker should send
            env outputs to.
        """
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout source ranks and sizes for receiving action chunks."""
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            dst_rank=self._rank,
        )

    def _init_env(self):
        if self.cfg.env.train.auto_reset:
            for i in range(self.stage_num):
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))

                if self.cfg.env.train.get("enable_offload", False) and hasattr(
                    self.env_list[i], "offload"
                ):
                    self.env_list[i].offload()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
        )
        return env_output, env_info

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"``, ``"eval"``, or ``"rewrite"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval", "rewrite"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(src_rank, self._rank, extra=mode),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    def recv_chunk_logprobs(self, input_channel: Channel, mode="train") -> torch.Tensor:
        """Receive and merge chunked logprobs for ReGRPO t_clip calculation.

        The method fetches one logprob shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env logprobs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated logprobs tensor with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        chunk_logprobs = []
        for src_rank, expected_size in src_ranks_and_sizes:
            logprob_i = input_channel.get(
                key=CommMapper.build_channel_key(src_rank, self._rank, extra=f"{mode}_logprobs"),
            )
            if not isinstance(logprob_i, torch.Tensor):
                logprob_i = torch.tensor(logprob_i)
            assert logprob_i.shape[0] == expected_size, (
                f"Expected logprob shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {logprob_i.shape}."
            )
            chunk_logprobs.append(logprob_i)
        chunk_logprobs = torch.cat(chunk_logprobs, dim=0)
        return chunk_logprobs

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    def split_env_batch(
        self,
        env_batch: dict[str, Any],
        sizes: list[int],
        mode: Literal["train", "eval"],
    ) -> list[dict[str, Any]]:
        """Split one env batch dict into size-specified sub-batches along dim-0.

        Tensor values are chunked on dim-0; list values are sliced proportionally;
        nested dict values are split recursively.

        Args:
            env_batch: Env output dictionary produced by ``EnvOutput.to_dict``.
            sizes: Batch sizes for each destination rank.
            mode: Rollout mode used for list-length validation.

        Returns:
            A list of split env batches, one item per destination rank.
        """
        count = len(sizes)
        total_size = sum(sizes)
        splitted_env_batches = [{} for _ in range(count)]
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == total_size, (
                    f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
                )
                splitted_values = torch.split(value, sizes, dim=0)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_values[i].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage} "
                        f"(train_num_envs_per_stage), got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage} "
                        f"(eval_num_envs_per_stage), got {length}"
                    )
                assert length == total_size, (
                    f"List field '{key}' expected length {total_size}, got {length}."
                )
                begin = 0
                for i, size in enumerate(sizes):
                    splitted_env_batches[i][key] = value[begin : begin + size]
                    begin += size
            elif isinstance(value, dict):
                splitted_sub_batches = self.split_env_batch(value, sizes, mode)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_sub_batches[i]
            else:
                for i in range(count):
                    splitted_env_batches[i][key] = value

        return splitted_env_batches

    def send_env_batch(
        self,
        output_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval", "rewrite"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            output_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"``, ``"eval"``, or ``"rewrite"``.
        """
        assert mode in ["train", "eval", "rewrite"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = self.split_env_batch(env_batch, split_sizes, mode)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            output_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=mode),
            )

    @Worker.timer("interact")
    def interact(self, input_channel: Channel, output_channel: Channel):
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)

        # ReGRPO: initialize buffers for chunk states and returns
        if self.enable_regrpo:
            self.chunk_states_buffer = [
                [] for _ in range(self.stage_num)
            ]  # [stage][epoch][chunk]
            self.chunk_actions_buffer = [
                [] for _ in range(self.stage_num)
            ]  # [stage][epoch][chunk]
            self.chunk_rewards_buffer = [
                [] for _ in range(self.stage_num)
            ]  # [stage][epoch][chunk]
            self.chunk_logprobs_buffer = [
                [] for _ in range(self.stage_num)
            ]  # [stage][epoch][chunk]
            self.chunk_termination_idx_buffer = [
                [] for _ in range(self.stage_num)
            ]  # [stage][epoch] -> tensor [num_envs]
            self.epoch_returns_buffer = [
                [] for _ in range(self.stage_num)
            ]  # [stage][epoch]
            # Track cumulative rewards per epoch per env
            epoch_cumulative_rewards = [
                torch.zeros(self.train_num_envs_per_stage)
                for _ in range(self.stage_num)
            ]

        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_output_list = []

            # ReGRPO: initialize chunk states and actions for this epoch
            if self.enable_regrpo:
                for stage_id in range(self.stage_num):
                    self.chunk_states_buffer[stage_id].append([])
                    self.chunk_actions_buffer[stage_id].append([])
                    self.chunk_rewards_buffer[stage_id].append([])
                    self.chunk_logprobs_buffer[stage_id].append([])
                    # Initialize termination index to n_chunk_steps (meaning no termination yet)
                    self.chunk_termination_idx_buffer[stage_id].append(
                        torch.full((self.train_num_envs_per_stage,), n_chunk_steps, dtype=torch.long)
                    )
                    epoch_cumulative_rewards[stage_id].zero_()

            if not self.cfg.env.train.auto_reset:
                for stage_id in range(self.stage_num):
                    self.env_list[stage_id].is_start = True
                    extracted_obs, infos = self.env_list[stage_id].reset()
                    dones = (
                        torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    terminations = dones.clone()
                    truncations = dones.clone()

                    env_output = EnvOutput(
                        obs=extracted_obs,
                        dones=dones,
                        terminations=terminations,
                        truncations=truncations,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                        intervene_actions=None,
                        intervene_flags=None,
                    )
                    env_output_list.append(env_output)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                dones = (
                    torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                    .unsqueeze(1)
                    .repeat(1, self.cfg.actor.model.num_action_chunks)
                )
                terminations = dones.clone()
                truncations = dones.clone()

                for stage_id in range(self.stage_num):
                    env_output = EnvOutput(
                        obs=self.last_obs_list[stage_id],
                        rewards=None,
                        dones=dones,
                        terminations=terminations,
                        truncations=truncations,
                        intervene_actions=self.last_intervened_info_list[stage_id][0],
                        intervene_flags=self.last_intervened_info_list[stage_id][1],
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_output_list[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for chunk_idx in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    # ReGRPO: save chunk state BEFORE executing action
                    if self.enable_regrpo and hasattr(
                        self.env_list[stage_id], "get_state"
                    ):
                        # BP1: 只在第一个 chunk 触发一次，检查 enable_regrpo=True, hasattr=True
                        if len(self.chunk_states_buffer[stage_id][epoch]) == 0:
                            pass
                        chunk_state = self.env_list[stage_id].get_state()
                        self.chunk_states_buffer[stage_id][epoch].append(chunk_state)

                    raw_chunk_actions = self.recv_chunk_actions(input_channel)

                    # ReGRPO: receive and save logprobs for t_clip calculation
                    if self.enable_regrpo:
                        chunk_logprobs = self.recv_chunk_logprobs(input_channel)
                        self.chunk_logprobs_buffer[stage_id][epoch].append(
                            chunk_logprobs.clone().cpu()
                        )

                    # ReGRPO: save actions for rewriting
                    if self.enable_regrpo:
                        if isinstance(raw_chunk_actions, torch.Tensor):
                            self.chunk_actions_buffer[stage_id][epoch].append(
                                raw_chunk_actions.clone().cpu()
                            )
                        else:
                            # numpy array
                            self.chunk_actions_buffer[stage_id][epoch].append(
                                torch.from_numpy(raw_chunk_actions.copy())
                            )

                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(output_channel, env_output.to_dict())
                    env_output_list[stage_id] = env_output

                    # ReGRPO: save per-chunk rewards and accumulate
                    if self.enable_regrpo and env_output.rewards is not None:
                        # rewards shape: [num_envs, num_action_chunks]
                        rewards = env_output.rewards
                        if isinstance(rewards, torch.Tensor):
                            chunk_reward_sum = rewards.sum(dim=-1).cpu()
                        else:
                            # numpy array
                            chunk_reward_sum = torch.from_numpy(
                                rewards.sum(axis=-1).copy()
                            )
                        self.chunk_rewards_buffer[stage_id][epoch].append(
                            chunk_reward_sum.clone()
                        )
                        epoch_cumulative_rewards[stage_id] += chunk_reward_sum

                    # ReGRPO: record termination position for each env
                    if self.enable_regrpo and env_output.dones is not None:
                        # dones shape: [num_envs, num_action_chunks]
                        # Check if any action in this chunk is done
                        chunk_done = env_output.dones.any(dim=-1)  # [num_envs]
                        # Update termination index: only record the first termination
                        # If env is done and termination_idx is still n_chunk_steps, set it to chunk_idx
                        termination_idx = self.chunk_termination_idx_buffer[stage_id][epoch]
                        not_terminated_yet = termination_idx >= n_chunk_steps
                        newly_terminated = chunk_done & not_terminated_yet
                        termination_idx[newly_terminated] = chunk_idx

                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)

            # ReGRPO: store epoch returns
            if self.enable_regrpo:
                for stage_id in range(self.stage_num):
                    self.epoch_returns_buffer[stage_id].append(
                        epoch_cumulative_rewards[stage_id].clone()
                    )

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_intervened_info_list = [
                (env_output.intervene_actions, env_output.intervene_flags)
                for env_output in env_output_list
            ]
            self.finish_rollout()
            # breakpoint()
        # ReGRPO: compute t_clip, perform rewriting, and calculate p_flip
        if self.enable_regrpo:
            # Save channel references for rewriting
            self._rewrite_input_channel = input_channel
            self._rewrite_output_channel = output_channel
            self._perform_regrpo_rewriting()

        for env in self.env_list:
            if self.cfg.env.train.get("enable_offload", False) and hasattr(
                env, "offload"
            ):
                env.offload()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        # ReGRPO/SeGRPO: include t_clip and p_flip in metrics for ActorWorker
        if self.enable_regrpo and self.regrpo_p_flip is not None:
            env_metrics["regrpo_t_clip"] = self.regrpo_t_clip.cpu()
            env_metrics["regrpo_p_flip"] = self.regrpo_p_flip.cpu()
        # SeGRPO: include mean rewrite rewards
        if self.enable_regrpo and self.segrpo_rewrite_rewards is not None:
            env_metrics["segrpo_rewrite_rewards"] = self.segrpo_rewrite_rewards.cpu()

        return env_metrics

    def rewrite_trajectories(
        self,
        t_clip_per_traj: torch.Tensor,
        per_stage_epoch_min_t_clip: torch.Tensor,
        input_channel: Channel,
        output_channel: Channel,
    ) -> torch.Tensor:
        """
        ReGRPO: Rewrite trajectories from t_clip position using policy re-inference.

        Instead of replaying saved actions, this method interacts with the rollout
        worker to get new actions from the policy for each step.

        Args:
            t_clip_per_traj: Rewrite position for each trajectory.
                Shape: [total_num_envs * rollout_epoch] flattened across all stages/epochs.
            per_stage_epoch_min_t_clip: Minimum t_clip per (stage, epoch), used as the
                actual restore point. Shape: [stage_num, rollout_epoch].
            input_channel: Channel to receive actions from rollout worker.
            output_channel: Channel to send observations to rollout worker.

        Returns:
            rewrite_returns: Returns from rewritten trajectories.
                Shape: [total_num_envs * rollout_epoch, num_rewrite]
        """
        if not self.enable_regrpo:
            raise RuntimeError("rewrite_trajectories called but ReGRPO is not enabled")

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        num_rewrite = self.regrpo_num_rewrite
        rollout_epoch = self.cfg.algorithm.rollout_epoch

        # Reshape t_clip to [stage_num, rollout_epoch, num_envs_per_stage]
        total_trajs = t_clip_per_traj.shape[0]
        envs_per_stage = self.train_num_envs_per_stage
        expected_total = self.stage_num * rollout_epoch * envs_per_stage
        assert total_trajs == expected_total, (
            f"t_clip shape mismatch: got {total_trajs}, expected {expected_total}"
        )

        t_clip_reshaped = t_clip_per_traj.view(
            self.stage_num, rollout_epoch, envs_per_stage
        )

        # Storage for rewrite returns: [stage, epoch, env, rewrite_idx]
        all_rewrite_returns = torch.zeros(
            self.stage_num, rollout_epoch, envs_per_stage, num_rewrite
        )

        # Helper function to get the innermost unwrapped env
        def get_base_env(env):
            while hasattr(env, 'env') and 'get_state' not in type(env).__dict__:
                env = env.env
            return env

        # Onload envs if needed
        for env in self.env_list:
            base_env = get_base_env(env)
            if hasattr(base_env, "onload"):
                base_env.onload()

        # For each rewrite iteration
        for rewrite_idx in range(num_rewrite):
            # For each epoch
            for epoch in range(rollout_epoch):
                # For each stage
                for stage_id in range(self.stage_num):
                    env = self.env_list[stage_id]
                    base_env = get_base_env(env)
                    chunk_states = self.chunk_states_buffer[stage_id][epoch]

                    # Use per-(stage, epoch) minimum t_clip as the actual rewrite start.
                    # This is the earliest "critical decision point" among all envs in this
                    # stage/epoch, so all envs are restored to the same snapshot that
                    # precedes their respective t_clip.
                    min_t_clip = int(per_stage_epoch_min_t_clip[stage_id, epoch].item())
                    # Restore state at min_t_clip position
                    if min_t_clip < len(chunk_states):
                        base_env.load_state(chunk_states[min_t_clip])

                    # Track rewards for this rewrite
                    rewrite_cumulative_rewards = torch.zeros(envs_per_stage)

                    # Get initial observation after restoring state
                    extracted_obs = base_env.wrap_obs()
                    # Rollout from min_t_clip to end using policy re-inference
                    for chunk_idx in range(min_t_clip, n_chunk_steps):
                        # Send observation to rollout worker
                        env_output = EnvOutput(
                            obs=extracted_obs,
                            dones=torch.zeros((envs_per_stage,), dtype=bool)
                            .unsqueeze(1)
                            .repeat(1, self.cfg.actor.model.num_action_chunks),
                            terminations=torch.zeros((envs_per_stage,), dtype=bool)
                            .unsqueeze(1)
                            .repeat(1, self.cfg.actor.model.num_action_chunks),
                            truncations=torch.zeros((envs_per_stage,), dtype=bool)
                            .unsqueeze(1)
                            .repeat(1, self.cfg.actor.model.num_action_chunks),
                            rewards=None,
                            final_obs=None,
                        )
                        self.send_env_batch(output_channel, env_output.to_dict(), mode="rewrite")

                        # Receive action from rollout worker
                        action = self.recv_chunk_actions(input_channel, mode="rewrite")
                        if isinstance(action, np.ndarray):
                            action = torch.from_numpy(action)
                        action = action.to(env.device)

                        # Execute action with policy-generated action
                        if hasattr(env, "chunk_step"):
                            obs_list, rewards, terminations, truncations, infos = (
                                env.chunk_step(action)
                            )
                            # rewards shape: [num_envs, chunk]
                            if isinstance(rewards, torch.Tensor):
                                chunk_reward_sum = rewards.sum(dim=-1).cpu()
                            else:
                                chunk_reward_sum = torch.from_numpy(
                                    rewards.sum(axis=-1).copy()
                                )
                            # Update observation for next step
                            extracted_obs = base_env.wrap_obs()
                        else:
                            # Fallback for non-chunk environments
                            obs, rewards, terminations, truncations, infos = env.step(
                                action
                            )
                            if isinstance(rewards, torch.Tensor):
                                chunk_reward_sum = rewards.cpu()
                            else:
                                chunk_reward_sum = torch.from_numpy(
                                    np.atleast_1d(rewards).copy()
                                )
                            extracted_obs = obs

                        rewrite_cumulative_rewards += chunk_reward_sum

                    # Store rewrite returns
                    all_rewrite_returns[stage_id, epoch, :, rewrite_idx] = (
                        rewrite_cumulative_rewards
                    )

        # Offload envs if needed
        for env in self.env_list:
            if self.cfg.env.train.get("enable_offload", False) and hasattr(
                env, "offload"
            ):
                env.offload()

        # Flatten returns to [total_trajs, num_rewrite]
        rewrite_returns = all_rewrite_returns.view(-1, num_rewrite)
        return rewrite_returns

    def get_original_returns(self) -> torch.Tensor:
        """
        ReGRPO: Get original trajectory returns from buffer.

        Returns:
            original_returns: Shape [total_num_envs * rollout_epoch]
        """
        if not self.enable_regrpo:
            raise RuntimeError("get_original_returns called but ReGRPO is not enabled")

        # epoch_returns_buffer[stage_id][epoch] = tensor [num_envs_per_stage]
        all_returns = []
        for stage_id in range(self.stage_num):
            for epoch in range(len(self.epoch_returns_buffer[stage_id])):
                all_returns.append(self.epoch_returns_buffer[stage_id][epoch])

        return torch.cat(all_returns, dim=0)

    def _perform_regrpo_rewriting(self):
        """
        ReGRPO: Compute t_clip based on logprobs, perform rewriting, and compute p_flip.

        This method:
        1. Computes t_clip using action-level logprobs (finding lowest logprob chunk)
        2. Performs trajectory rewriting from t_clip positions
        3. Computes flip rate by comparing original and rewritten returns
        4. Stores t_clip and p_flip for later use by ActorWorker
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        envs_per_stage = self.train_num_envs_per_stage
        num_action_chunks = self.cfg.actor.model.num_action_chunks
        action_dim = self.cfg.actor.model.action_dim

        # 1. Compute t_clip based on action-level logprobs
        # Find the chunk with lowest logprob as the critical decision point
        all_t_clips = []
        for stage_id in range(self.stage_num):
            for epoch in range(rollout_epoch):
                chunk_logprobs = self.chunk_logprobs_buffer[stage_id][epoch]
                if len(chunk_logprobs) < 2:
                    # Not enough chunks, use middle point
                    t_clip = torch.full(
                        (envs_per_stage,), n_chunk_steps // 2, dtype=torch.long
                    )
                else:
                    # Stack logprobs: [n_chunks, num_envs, num_action_chunks * action_dim]
                    # or [n_chunks, num_envs, num_action_chunks, action_dim]
                    logprobs_stacked = torch.stack(chunk_logprobs, dim=0)

                    # Compute action-level logprobs (sum over action dimensions)
                    # logprobs shape: [n_chunks, num_envs, num_action_chunks * action_dim]
                    # -> [n_chunks, num_envs, num_action_chunks] -> [n_chunks, num_envs]
                    if logprobs_stacked.dim() == 4:
                        # [n_chunks, num_envs, num_action_chunks, action_dim]
                        action_logprobs = logprobs_stacked.sum(dim=-1)  # [n_chunks, num_envs, num_action_chunks]
                        chunk_level_logprobs = action_logprobs.sum(dim=-1)  # [n_chunks, num_envs]
                    else:
                        # [n_chunks, num_envs, num_action_chunks * action_dim]
                        # Reshape to [n_chunks, num_envs, num_action_chunks, action_dim]
                        n_chunks = logprobs_stacked.shape[0]
                        num_envs = logprobs_stacked.shape[1]
                        logprobs_reshaped = logprobs_stacked.view(
                            n_chunks, num_envs, num_action_chunks, action_dim
                        )
                        action_logprobs = logprobs_reshaped.sum(dim=-1)  # [n_chunks, num_envs, num_action_chunks]
                        chunk_level_logprobs = action_logprobs.sum(dim=-1)  # [n_chunks, num_envs]

                    # Get termination indices for this epoch
                    termination_idx = self.chunk_termination_idx_buffer[stage_id][epoch]  # [num_envs]

                    n_chunks = chunk_level_logprobs.shape[0]
                    num_envs = chunk_level_logprobs.shape[1]
                    device = chunk_level_logprobs.device

                    # Compute sliding window sum of 3 consecutive logprobs
                    # For position i, sum logprobs[i] + logprobs[i+1] + logprobs[i+2]
                    window_size = 3
                    if n_chunks >= window_size:
                        # Create sliding window sums: [n_chunks - 2, num_envs]
                        window_sums = (
                            chunk_level_logprobs[:-2, :] +
                            chunk_level_logprobs[1:-1, :] +
                            chunk_level_logprobs[2:, :]
                        )  # [n_chunks - 2, num_envs]

                        # Create mask for valid window positions
                        # A window starting at position i is valid if all 3 chunks (i, i+1, i+2) are before termination
                        # i.e., i+2 < termination_idx, which means i < termination_idx - 2
                        n_windows = window_sums.shape[0]
                        window_indices = torch.arange(n_windows, device=device).unsqueeze(1)
                        termination_idx_expanded = termination_idx.unsqueeze(0).to(device)

                        # valid_window_mask[i, j] = True if window starting at i is valid for env j
                        # Window at position i covers chunks i, i+1, i+2, so need i+2 < termination_idx
                        valid_window_mask = (window_indices + window_size - 1) < termination_idx_expanded  # [n_windows, num_envs]

                        # Set invalid window sums to +inf so they won't be selected by argmin
                        masked_window_sums = window_sums.clone()
                        masked_window_sums[~valid_window_mask] = float('inf')

                        # Find the window with minimum sum for each env
                        # t_clip is the starting position of this window
                        t_clip = masked_window_sums.argmin(dim=0)  # [num_envs]
                    else:
                        # Not enough chunks for sliding window, use first valid chunk
                        t_clip = torch.zeros(num_envs, dtype=torch.long, device=device)

                    # Ensure at least min_prefix_chunks prefix chunks
                    # Also clamp to valid range considering window size
                    max_valid_t_clip = (termination_idx - window_size).clamp(min=self.regrpo_min_prefix_chunks)
                    clamp_max = n_chunk_steps - window_size
                    if self.regrpo_max_prefix_chunks is not None:
                        clamp_max = min(clamp_max, self.regrpo_max_prefix_chunks)
                    t_clip = t_clip.clamp(
                        min=self.regrpo_min_prefix_chunks,
                        max=clamp_max
                    )
                    t_clip = torch.min(t_clip, max_valid_t_clip)
                all_t_clips.append(t_clip)

        # Flatten t_clip: [total_trajs]
        self.regrpo_t_clip = torch.cat(all_t_clips, dim=0)

        # 2. Compute per-(stage, epoch) minimum t_clip as the actual rewrite start point.
        # Using the minimum across envs within each (stage, epoch) ensures all envs in a
        # stage are restored to the same snapshot that precedes every env's t_clip.
        t_clip_reshaped = self.regrpo_t_clip.view(
            self.stage_num, rollout_epoch, envs_per_stage
        )
        per_stage_epoch_min_t_clip = (
            t_clip_reshaped.min(dim=-1).values  # [stage_num, rollout_epoch]
            .clamp(min=self.regrpo_min_prefix_chunks)
            .long()
            .cpu()
        )

        # Send the per-(stage, epoch) min t_clips to each rollout worker so it can
        # determine how many rewrite steps to generate for each (stage, epoch).
        for dst_rank, _ in self.dst_ranks["rewrite"]:
            self._rewrite_output_channel.put(
                item=per_stage_epoch_min_t_clip,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra="rewrite_meta"
                ),
            )

        # 3. Perform rewriting with policy re-inference
        rewrite_returns = self.rewrite_trajectories(
            self.regrpo_t_clip,
            per_stage_epoch_min_t_clip=per_stage_epoch_min_t_clip,
            input_channel=self._rewrite_input_channel,
            output_channel=self._rewrite_output_channel,
        )
        # rewrite_returns shape: [total_trajs, num_rewrite]

        # Flush rewrite frames into a separate "rewrite" sub-directory so they
        # do not leak into the next rollout's video.
        for i in range(self.stage_num):
            if self.cfg.env.train.video_cfg.save_video and isinstance(
                self.env_list[i], RecordVideo
            ):
                self.env_list[i].flush_video(video_sub_dir="rewrite")

        # 3. Get original returns
        original_returns = self.get_original_returns()
        # original_returns shape: [total_trajs]

        # 4. Compute flip rate
        success_threshold = self.regrpo_success_threshold
        orig_success = original_returns > success_threshold  # [total_trajs]
        rewrite_success = rewrite_returns > success_threshold  # [total_trajs, num_rewrite]

        # Count flips
        flips = (rewrite_success != orig_success.unsqueeze(1)).float()
        flip_count = flips.sum(dim=1)  # [total_trajs]
        num_rewrite = rewrite_returns.shape[1]

        self.regrpo_p_flip = flip_count / num_rewrite
        # Clamp: min is 1/num_rewrite (statistical minimum: at least 1 flip out of num_rewrite)
        # Using 0 would cause division issues; using values below 1/num_rewrite has no statistical basis.
        self.regrpo_p_flip = self.regrpo_p_flip.clamp(min=0.0, max=0.99)

        # SeGRPO: store mean rewrite reward per trajectory
        self.segrpo_rewrite_rewards = rewrite_returns.mean(dim=1)  # [total_trajs]

        # Debug: check flip rate and returns
        # breakpoint()
        # 可查看的变量:
        # - original_returns: 每个 traj 的原始 return [total_trajs]
        # - rewrite_returns: 重写后的 return [total_trajs, num_rewrite]
        # - self.regrpo_p_flip: 翻转率 [total_trajs]
        # - orig_success: 原始是否成功 [total_trajs]
        # - rewrite_success: 重写是否成功 [total_trajs, num_rewrite]
        # - self.regrpo_t_clip: 每个 traj 的 t_clip 位置 [total_trajs]

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        eval_metrics = defaultdict(list)

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            for stage_id in range(self.stage_num):
                self.eval_env_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_env_list[stage_id].reset()
                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                )
                self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

            for eval_step in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)
                    if eval_step == n_chunk_steps - 1:
                        continue
                    self.send_env_batch(
                        output_channel, env_output.to_dict(), mode="eval"
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
