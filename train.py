#!/usr/bin/env python
import sys
import os

import wandb

import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import yaml
from configs.config import get_config
from environment.env_wrappers import SubprocVecEnv
from runner.shared_runner import SharedRunner

import algorithm.policy
import environment.env_base as env_base


def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    # TODO: register yaml file
    with open(os.path.join('configs', all_args.config + ".yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.get("base", {}).items():
        setattr(all_args, k, v)
    policy_config = config['policy']
    environment_config = config['environment']
    all_args.env_name = environment_config['type']

    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed = all_args.runing_id
    else:
        all_args.seed = np.random.randint(1000, 10000)
    print("seed is :", all_args.seed)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
        "/results") / all_args.env_name / all_args.experiment_name / str(
            all_args.seed)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # TODO: control by config directly
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         group=all_args.experiment_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.experiment_name) + "_seed" +
                         str(all_args.seed),
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

    setproctitle.setproctitle(
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" +
        str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = SubprocVecEnv([
        lambda: env_base.make(environment_config, split='train')
        for rank in range(all_args.n_rollout_threads)
    ])
    if all_args.use_render:
        eval_envs = SubprocVecEnv([
            lambda: env_base.make(environment_config, split='render')
            for rank in range(all_args.n_render_rollout_threads)
        ])
    elif all_args.use_eval:
        eval_envs = SubprocVecEnv([
            lambda: env_base.make(environment_config, split='eval')
            for rank in range(all_args.n_eval_rollout_threads)
        ])
    else:
        eval_envs = None
    num_agents = envs.num_agents
    all_args.num_agents = num_agents

    policy = algorithm.policy.make(policy_config, envs.observation_spaces[0],
                                   envs.action_spaces[0])

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    # run experiments
    runner = SharedRunner(config, policy)
    if not all_args.use_render:
        runner.run()
    else:
        assert all_args.model_dir is not None
        assert all_args.n_render_rollout_threads == 1
        runner.eval(0, render=True)

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        if hasattr(runner, writer):
            runner.writer.export_scalars_to_json(
                str(runner.log_dir + '/summary.json'))
            runner.writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
