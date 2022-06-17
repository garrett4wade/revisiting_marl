import socket
from absl import flags

FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

import environment.env_base
import environment.env_utils

import environment.football.football_env
