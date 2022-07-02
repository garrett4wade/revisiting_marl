#!/usr/bin/env python
from pathlib import Path
import gym
import itertools
import logging
import multiprocessing as mp
import numpy as np
import os
import setproctitle
import socket
import sys
import torch
import wandb
import yaml

from algorithm.trainer import SampleBatch
from configs.config import get_base_config, make_config
from environment.env_wrappers import shared_env_worker, shared_eval_worker, EnvironmentControl, TorchTensorWrapper
from runner.shared_runner import SharedRunner
from utils.namedarray import recursive_apply
import algorithm.policy
import environment.env_base as env_base

logging.basicConfig(
    format=
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('log.txt', mode='w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def main(args):
    parser = get_base_config()
    all_args = parser.parse_known_args(args)[0]
    config = make_config(all_args.config)
    for k, v in config.get("base", {}).items():
        if f"--{k}" not in args:
            setattr(all_args, k, v)
        else:
            logger.warning(f"CLI argument {k} conflicts with yaml config. "
                           f"The latter will be overwritten "
                           f"by CLI arguments {k}={getattr(all_args, k)}.")

    policy_config = config['policy']
    environment_config = config['environment']
    all_args.env_name = environment_config['type']

    logger.info("all config: {}".format(all_args))
    if all_args.seed_specify:
        all_args.seed = all_args.runing_id
    else:
        all_args.seed = np.random.randint(1000, 10000)
    logger.info("seed is: {}".format(all_args.seed))
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logger.info("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        logger.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    if not all_args.eval:
        run_dir = (Path("results") / all_args.env_name /
                   all_args.experiment_name / str(all_args.seed))
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        if all_args.use_wandb:
            run = wandb.init(
                config=all_args,
                project=all_args.wandb_project
                if all_args.wandb_project else all_args.env_name,
                group=all_args.wandb_group
                if all_args.wandb_group else all_args.experiment_name,
                entity=all_args.user_name,
                notes=socket.gethostname(),
                name=all_args.wandb_name if all_args.wandb_name else
                str(all_args.experiment_name) + "_seed" + str(all_args.seed),
                dir=str(run_dir),
                job_type="training",
                reinit=True)
        else:
            if not run_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [
                    int(str(folder.name).split('run')[1])
                    for folder in run_dir.iterdir()
                    if str(folder.name).startswith('run')
                ]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))
    else:
        run_dir = None

    setproctitle.setproctitle(
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" +
        str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    example_env = TorchTensorWrapper(
        env_base.make(environment_config, split='train'), device)
    act_space = example_env.action_spaces[0]
    obs_space = example_env.observation_spaces[0]
    all_args.num_agents = num_agents = example_env.num_agents
    del example_env

    policy = algorithm.policy.make(policy_config, obs_space, act_space)

    if isinstance(act_space, gym.spaces.Discrete):
        act_dim = 1
    elif isinstance(act_space, gym.spaces.Box):
        act_dim = act_space.shape[0]
    elif isinstance(act_space, gym.spaces.MultiDiscrete):
        act_dim = act_space.nvec
    else:
        raise NotImplementedError()

    # initialze storage
    storage = SampleBatch(
        # NOTE: sampled available actions should be 1
        obs=obs_space.sample(),
        value_preds=torch.zeros(1),
        actions=torch.zeros(act_dim),
        action_log_probs=torch.zeros(1),
        rewards=torch.zeros(1),
        masks=torch.ones(1),
        active_masks=torch.ones(1),
        bad_masks=torch.ones(1),
    )

    if policy.num_rnn_layers > 0:
        storage.policy_state = policy.policy_state_space.sample()

    storage = recursive_apply(
        storage,
        lambda x: x.repeat(all_args.episode_length + 1, all_args.
                           n_rollout_threads, num_agents,
                           *((1, ) * len(x.shape))).share_memory_(),
    )
    storage.step = torch.tensor(0, dtype=torch.long).share_memory_()

    eval_storage = SampleBatch(
        obs=obs_space.sample(),
        masks=torch.ones(1),
        actions=torch.zeros(act_dim),
        value_preds=None,
        action_log_probs=None,
        rewards=None,
        active_masks=None,
        bad_masks=None,
    )
    if policy.num_rnn_layers > 0:
        eval_storage.policy_state = policy.policy_state_space.sample()
    eval_storage = recursive_apply(
        eval_storage,
        lambda x: x.repeat(all_args.n_eval_rollout_threads, num_agents,
                           *((1, ) * len(x.shape))).share_memory_(),
    )

    # initialize communication utilities
    env_ctrls = [
        EnvironmentControl(mp.Semaphore(0), mp.Semaphore(0), mp.Event())
        for _ in range(all_args.n_rollout_threads)
    ]
    eval_env_ctrls = [
        EnvironmentControl(mp.Semaphore(0), mp.Semaphore(0), mp.Event(),
                           mp.Event(), mp.Event())
        for _ in range(all_args.n_eval_rollout_threads)
    ]
    info_queue = mp.Queue(1000)
    eval_info_queue = mp.Queue(all_args.n_eval_rollout_threads)

    # start worker
    # TODO: config env number
    env_workers = [
        mp.Process(
            target=shared_env_worker,
            args=(i, [environment_config], env_ctrls[i], storage, info_queue),
        ) for i in range(all_args.n_rollout_threads)
    ]
    for worker in env_workers:
        worker.start()

    eval_workers = [
        mp.Process(
            target=shared_eval_worker,
            args=(
                i,
                [environment_config],
                eval_env_ctrls[i],
                eval_storage,
                eval_info_queue,
            ),
            kwargs=dict(render=all_args.render),
        ) for i in range(all_args.n_eval_rollout_threads)
    ]
    for ew in eval_workers:
        ew.start()

    # run experiments
    runner = SharedRunner(all_args,
                          policy,
                          storage,
                          env_ctrls,
                          info_queue,
                          eval_storage,
                          eval_env_ctrls,
                          eval_info_queue,
                          device,
                          run_dir=run_dir)

    if all_args.eval:
        assert all_args.model_dir is not None
        if all_args.render:
            assert all_args.n_eval_rollout_threads == 1
        runner.eval(0)
    else:
        runner.run()

    # post process
    for ctrl in itertools.chain(env_ctrls, eval_env_ctrls):
        ctrl.exit_.set()
    for worker in itertools.chain(env_workers, eval_workers):
        worker.join()

    if not all_args.eval:
        if all_args.use_wandb:
            run.finish()
        else:
            if hasattr(runner, "writer"):
                runner.writer.export_scalars_to_json(
                    str(runner.log_dir + '/summary.json'))
                runner.writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
