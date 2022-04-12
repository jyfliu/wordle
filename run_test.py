from game import *
from easydict import EasyDict as edict
import train
import torch
import model

# config = edict({
  # 'batch_size': 1,
  # 'n_letters': 5,
  # 'n_guesses': 8,
  # 'path': 'res/official',
  # 'hidden': 32,
  # 'att_hidden': 16,
  # 'device': 'cpu',

  # 'epsilon_start': 1,
  # 'epsilon_end': 0.1,
  # 'steps_max': 200,
  # 'n_episodes': 2000,
  # 'train_after_episodes': 10,
  # 'train_epochs': 25,
  # 'test_epochs': 10,

  # 'lr': 5e-4,
  # 'gamma': 0.995,
  # 'target_network_update_freq': 10,

  # 'seed': 1,
  # 'experiment': 'with_dict',
  # 'dict_n_words': 12972,
# })

# with open(f"{config.path}_answer.txt", "r") as f:
  # answers = f.read().splitlines()

# with open(f"{config.path}_guess.txt", "r") as f:
  # guesses = f.read().splitlines()
# config.act_n = len(answers) + len(guesses)
# config.choice_n = 1
# config.obs_n = config.act_n + config.n_letters * 2
# config.state_n = 1 + config.n_letters * 2

# q = torch.load("data/three_policy_fixed_small_DRQN_weights_epi_5000.pt")
q = torch.load("final_data/REINFORCE_baseline_model.pt")
config = q.config
print(config)
config.batch_size = 100
# q = model.DRQN(config)
env = train.Environment(config)

# q.load_state_dict(torch.load("data/DRQN_weights_epi_175.pt"))

_, _, R, _ = env.play_episode_rb(q.test_policy)
print("Mean rwd", (sum(R)/ len(R)).mean().item())
print("Avg num guesses", env.wordle.total_num_guesses / 100.)
tmp = env.wordle.batch_num_guesses * (1-env.wordle.game_not_done)
print("Avg winning num guess", tmp.sum().item() / (1-env.wordle.game_not_done).sum().item())
print("Winning pct", 1-env.wordle.game_not_done.sum().item() / 100.)

