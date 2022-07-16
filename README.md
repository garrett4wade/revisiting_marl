## Introduction

Official codebase for paper "[Revisiting Some Common Practices in Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2206.07505)" (ICML22).

## Content
+ `python perm_qlearning.py`: VD results in the XOR game (Figure 1 in the paper).
+ `python perm_ar.py`: AR results in the 4-player permutation game (Figure 2, 3 in the paper).
+ Run auto-regressive policy learning in the Bridge game: `python main.py --config bridge`.
  + Run `python main.py --config bridge --render --render_mode rgb_array --save_video --video_fps 5 --model_dir results/bridge/check/1/run1/models --video_file output.gif --eval_episodes 5` to see GIF deomstration of the learned multi-modal behavior.
  + The attention-based policy is implemented in `algorithm/modules/ar_utils.py` and `algorithm/policies/bridge_ar.py`. Similar policies are implemented for SMAC and GRF.
  + The multi-step optimization technique is implemented in `algorithm/trainers/ar_mappo.py`.
  + The random order technique is implemented in `algorithm/policies/ar_policy_base.py`.
+ Experiments conducted in SMAC and GRF directly adopt the [MAPPO codebase](https://github.com/marlbenchmark/on-policy).
+ Experiments regarding HAPPO adopt [this forked repo](https://github.com/garrett4wade/Trust-Region-Methods-in-Multi-Agent-Reinforcement-Learning).

