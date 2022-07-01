from collections import defaultdict
import math
import torch
import torch.nn as nn

from algorithm.trainer import SampleBatch, feed_forward_generator, recurrent_generator


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm()**2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


class MAPPO:

    def __init__(self, args, policy):
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=args.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=args.critic_lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )

    def cal_value_loss(self, values, value_preds_batch, return_batch,
                       active_masks_batch):
        if self.policy.popart_head is not None:
            self.policy.update_popart(return_batch)
            return_batch = self.policy.normalize_value(return_batch)

        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch).clamp(-self.clip_param,
                                              self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss *
                          active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample: SampleBatch, update_actor=True):
        # Reshape to do in a single forward pass for all steps
        action_log_probs, values, dist_entropy = self.policy.analyze(sample)
        # actor update
        imp_weights = torch.exp(action_log_probs - sample.action_log_probs)

        surr1 = imp_weights * sample.advantages
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * sample.advantages
        assert surr1.shape[-1] == surr2.shape[-1] == 1

        if self._use_policy_active_masks:
            policy_loss = (-torch.min(surr1, surr2) * sample.active_masks
                           ).sum() / sample.active_masks.sum()
            dist_entropy = (dist_entropy * sample.active_masks
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

    def train(self, storage, update_actor=True):
        train_info = defaultdict(lambda: 0)

        for _ in range(self.ppo_epoch):
            if self.policy.num_rnn_layers > 0:
                data_generator = recurrent_generator(storage,
                                                     self.num_mini_batch,
                                                     self.data_chunk_length)
            else:
                data_generator = feed_forward_generator(
                    storage, self.num_mini_batch)

            for sample in data_generator:
                (value_loss, critic_grad_norm, policy_loss, dist_entropy,
                 actor_grad_norm,
                 imp_weights) = self.ppo_update(sample,
                                                update_actor=update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
