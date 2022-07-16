from typing import Optional, List, Dict
import dataclasses
import numpy as np
import time
import torch.nn as nn
import torch

from algorithm.trainers.mappo import MAPPO


class AutoRegressiveMAPPO(MAPPO):

    def ppo_update(self, sample, update_actor=True):

        (action_log_probs, values, dist_entropy, joint_entropy,
         all_execution_mask) = self.policy.analyze(sample)

        raw_importance_ratio = action_log_probs - sample.action_log_probs
        raw_importance_ratio *= sample.masks  # [T, bs, n_agents, 1]
        all_execution_mask = all_execution_mask.unsqueeze(-1)
        importance_ratio = [
            (raw_importance_ratio * all_execution_mask[..., i, :, :]).sum(
                -2, keepdim=True) for i in range(self.policy.n_agents)
        ]
        imp_weights = torch.cat(importance_ratio, -2).exp()

        surr1 = imp_weights * sample.advantages
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * sample.advantages
        assert surr1.shape[-1] == surr2.shape[-1] == 1

        if self._use_policy_active_masks:
            policy_loss = (-torch.min(surr1, surr2) * sample.active_masks
                           ).sum() / sample.active_masks.sum()
            dist_entropy = (dist_entropy * sample.active_masks.squeeze(-1)
                            ).sum() / sample.active_masks.sum()
        else:
            policy_loss = -torch.min(surr1, surr2).mean()
            dist_entropy = dist_entropy.mean()

        value_loss = self.cal_value_loss(values, sample.value_preds,
                                         sample.returns, sample.active_masks)

        self.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights
