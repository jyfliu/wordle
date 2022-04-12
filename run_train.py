from game import *
from easydict import EasyDict as edict
import train
import torch

torch.manual_seed(1)

# train.Trainer(edict({
  # 'batch_size': 50,
  # 'n_letters': 5,
  # 'n_guesses': 8,
  # 'path': 'res/official',
  # 'hidden': 32,
  # 'att_hidden': 16,
  # 'device': 'cpu',

  # 'epsilon_start': 1,
  # 'epsilon_end': 0.1,
  # 'steps_max': 1000,
  # 'n_episodes': 2000,
  # 'train_after_episodes': 10,
  # 'train_epochs': 25,
  # 'test_epochs': 1,

  # 'model': 'attention', # [rnn, attention]
  # 'action_rep': 'score', # [onehot, score]
  # 'algo': 'Q', # [Q, policy]
  # 'lr': 5e-3,
  # 'gamma': 0.9,
  # # q learning hyperparams
  # 'target_network_update_freq': 10,
  # # policy gradient hyperparams
  # 'policy_train_iters': 1,

  # 'seed': 1,
  # 'experiment': 'with_dict', # [with_dict, no_information]
  # 'dict_n_words': 12972,
  # 'log_path': ''
# })).train()

# train.Trainer(edict({
  # 'batch_size': 50,
  # 'n_letters': 5,
  # 'n_guesses': 8,
  # 'path': 'res/official',
  # 'hidden': 32,
  # 'att_hidden': 16,
  # 'device': 'cpu',

  # 'epsilon_start': 1,
  # 'epsilon_end': 0.1,
  # 'steps_max': 1000,
  # 'n_episodes': 2000,
  # 'train_after_episodes': 0,
  # 'train_epochs': 1,
  # 'test_epochs': 1,

  # 'model': 'attention', # [rnn, attention]
  # 'action_rep': 'score', # [onehot, score]
  # 'algo': 'policy_baseline', # [Q, policy]
  # 'lr': 5e-3,
  # 'gamma': 0.9,
  # # q learning hyperparams
  # 'target_network_update_freq': 10,
  # # policy gradient hyperparams
  # 'policy_train_iters': 25,

  # 'seed': 1,
  # 'experiment': 'with_dict', # [with_dict, no_information]
  # 'dict_n_words': 12972,
  # 'log_path': 'policy_baseline_att_score',
# })).train()

train.Trainer(edict({
  'batch_size': 50,
  'n_letters': 3,
  'n_guesses': 6,
  'alphabet': 'reduced',
  'n_alphabet': 9,
  'path': 'res/three',
  'hidden': 32,
  'att_hidden': 16,
  'device': 'cpu',

  'epsilon_start': 1,
  'epsilon_end': 0.1,
  'steps_max': 1000,
  'n_episodes': 2000,
  'train_after_episodes': 10,
  'train_epochs': 25,
  'test_epochs': 1,

  'model': 'attention', # [rnn, attention]
  'action_rep': 'score', # [onehot, score]
  'algo': 'policy', # [Q, policy, ppo] (policy is reinforce+baseline)
  'lr': 5e-3,
  'gamma': 0.9,
  # q learning hyperparams
  'target_network_update_freq': 10,
  # policy gradient hyperparams
  'policy_train_iters': 25,

  'seed': 1,
  'experiment': 'with_dict', # [with_dict, no_information]
  'dict_n_words': 110,
  'log_path': 'three_Q',
  'continue_path': '', # initialize weights with model snapshot
})).train()


