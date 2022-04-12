import torch
import numpy as np
import random
import utils
import model
import game
import copy
import tqdm

class ReplayBuffer:

  def __init__(self, config):
    self.config = config
    self.device = config.device

    self.states = torch.empty((0, config.n_guesses + 1, config.state_n))
    if self.config.experiment == 'no_knowledge':
      self.actions = torch.empty((0, config.n_guesses, config.n_letters))
    elif self.config.experiment == 'with_dict':
      self.actions = torch.empty((0, config.n_guesses))
    self.rewards = torch.empty((0, config.n_guesses))
    self.not_done = torch.empty((0, config.n_guesses))

  def insert(self, states, actions, rewards, not_done):
    states = torch.stack(states).transpose(0, 1).to("cpu")
    actions = torch.stack(actions).transpose(0, 1).to("cpu")
    rewards = torch.stack(rewards).transpose(0, 1).to("cpu")
    not_done = torch.stack(not_done).transpose(0, 1).to("cpu")

    self.states = torch.cat([self.states, states], dim=0)
    self.actions = torch.cat([self.actions, actions], dim=0)
    self.rewards = torch.cat([self.rewards, rewards], dim=0)
    self.not_done = torch.cat([self.not_done, not_done], dim=0)

  def begin_state(self):
    return torch.zeros((self.config.batch_size, self.config.state_n), device=self.device)

  def state_to_onehot(self, state):
    # takes either a single step (in which case dim=2) or a complete episode (in which case dim=3)
    dim = len(state.shape)
    if dim == 2:
      state = state.unsqueeze(0)

    n_guess, bs, n_choice = state.shape

    action = state[:, :, :self.config.n_letters].long()
    state = state[:, :, self.config.n_letters:]

    action_onehot = torch.zeros((n_guess, bs, self.config.n_letters, self.config.n_alphabet), device=action.device)
    action_onehot.scatter_(3, action.unsqueeze(3), 1)
    action_onehot = action_onehot.view(n_guess, bs, self.config.n_letters * self.config.n_alphabet)
    if dim == 3:
      # in this case we want to set the first state to be the initial state of all zeros
      action_onehot[0] = 0
    state = torch.cat([action_onehot, state], dim=2)
    if dim == 2:
      state = state.squeeze(0)
    return state

  def sample_batch(self):
    idx = random.sample(range(self.states.shape[0]), self.config.batch_size)
    idx = torch.tensor(idx)
    state = self.states[idx].transpose(0, 1) # n_guess x B x (num types of actions)
    state = self.state_to_onehot(state)
    return (
      state.to(self.device), # n_guess x B x (n_action + n_letters*2)
      self.actions[idx].transpose(0, 1).to(self.device), # n_guess x B x n_choice
      self.rewards[idx].transpose(0, 1).to(self.device), # n_guess x B
      self.not_done[idx].transpose(0, 1).to(self.device), # n_guess x B
    )


class Environment:

  def __init__(self, config):
    self.config = config

    self.wordle = game.Wordle(config)
    self.replay = ReplayBuffer(config)

  def play_episode(self, policy):
    states, actions, rewards, not_done = [], [], [], []
    states.append(self.replay.begin_state())
    self.wordle.new_game()

    hidden = None
    for i in range(self.config.n_guesses):
      game_not_done = self.wordle.game_not_done.clone()
      if i == 0:
        obs = torch.zeros((self.config.batch_size, self.config.obs_n), device=self.config.device)
      else:
        obs = self.replay.state_to_onehot(states[-1])
      action, hidden = policy(obs, hidden)

      if self.config.experiment == 'no_knowledge':
        guess = action
      elif self.config.experiment == 'with_dict':
        guess = self.wordle.guesses[action]

      reward, green, yello = self.wordle.guess(guess)

      actions.append(action)
      rewards.append(reward)
      states.append(torch.cat([guess, green, yello], dim=1))
      not_done.append(game_not_done)

    return states, actions, rewards, not_done

  def play_episode_rb(self, policy):
    states, actions, rewards, not_done = [], [], [], []
    states.append(self.replay.begin_state())
    self.wordle.new_game()

    hidden = None
    for i in range(self.config.n_guesses):
      game_not_done = self.wordle.game_not_done.clone()
      if i == 0:
        obs = torch.zeros((self.config.batch_size, self.config.obs_n), device=self.config.device)
      else:
        obs = self.replay.state_to_onehot(states[-1])
      action, hidden = policy(obs, hidden)

      if self.config.experiment == 'no_knowledge':
        guess = action
      elif self.config.experiment == 'with_dict':
        guess = self.wordle.guesses[action]
      # guess = B x num_letters

      reward, green, yello = self.wordle.guess(guess)

      actions.append(action)
      rewards.append(reward)
      states.append(torch.cat([guess, green, yello], dim=1))
      not_done.append(game_not_done)

    self.replay.insert(states, actions, rewards, not_done)
    return states, actions, rewards, not_done


class Trainer:

  def __init__(self, config):
    self.config = config

    if config.experiment == 'no_knowledge':
      config.act_n = config.n_letters * self.config.n_alphabet
      config.choice_n = config.n_letters
      #                 last guess       + green + yello
      config.obs_n = config.act_n + config.n_letters * 2
      config.state_n = config.n_letters + config.n_letters * 2
    elif config.experiment == 'with_dict':
      with open(f"{config.path}_answer.txt", "r") as f:
        answers = f.read().splitlines()

      with open(f"{config.path}_guess.txt", "r") as f:
        guesses = f.read().splitlines()
      config.act_n = len(answers) + len(guesses)
      config.choice_n = 1
      config.obs_n = config.n_letters * self.config.n_alphabet + config.n_letters * 2
      config.state_n = config.n_letters + config.n_letters * 2

    utils.seed(config.seed)
    self.wordle = game.Wordle(config)
    self.env = Environment(config)

    if self.config.algo == "Q":
      if config.continue_path:
        # trust that the loaded model is compatible
        self.Q = torch.load(config.continue_path)
        self.Q.config = config
        self.Qt = torch.load(config.continue_path)
        self.Qt.config = config
      else:
        self.Q = model.DRQN(config).to(config.device)
        self.Qt = model.DRQN(config).to(config.device)
      self.optim = torch.optim.Adam(self.Q.parameters(), lr=self.config.lr)
    elif 'policy' in self.config.algo:
      if config.continue_path:
        # trust that the loaded model is compatible
        self.pi = torch.load(config.continue_path)
        self.pi.config = config
      else:
        self.pi = model.DRQN(config).to(config.device)
      self.optim = torch.optim.Adam(self.pi.parameters(), lr=self.config.lr)
      self.baseline = 0

  def update_networks(self, *args, **kwargs):
    if self.config.algo == "Q":
      return self.update_networks_Q(*args, **kwargs)
    elif 'policy' in self.config.algo:
      return self.update_networks_policy(*args, **kwargs)

  def update_networks_Q(self, episode):
    Q = self.Q
    Q_t = self.Q
    optim = self.optim

    S, A, R, D = self.env.replay.sample_batch()
    S = S.to(self.config.device)
    A = A.to(self.config.device)
    R = R.to(self.config.device)
    D = D.to(self.config.device)
    n_guess, B, _ = S.shape
    n_guess -= 1
    n_letter = self.config.n_letters

    if self.config.experiment == 'no_knowledge':
      QS, _ = Q(S) # QS = L + 1 x B x n_letters * 28
      QS = QS[:-1]
      QS = QS.view(n_guess * B * n_letter, self.config.n_alphabet)
      action = A.reshape(n_guess * B * n_letter, 1).long()

      qvalues = QS.gather(1, action).squeeze()
      qvalues = qvalues.view(n_guess * B, n_letter)

      with torch.no_grad():
        QS_t, _ = Q_t(S)
        QS_t = QS_t[1:]
        QS_t = QS_t.view(n_guess * B, n_letter, self.config.n_alphabet)
        q2values = torch.max(QS_t, dim = 2).values

      targets = R.reshape(n_guess * B, 1) + self.config.gamma * q2values

      loss = torch.nn.MSELoss()(targets, qvalues) * D.reshape(n_guess * B)
    elif self.config.experiment == 'with_dict':
      QS, _ = Q(S)
      QS = QS[:-1]
      QS = QS.view(n_guess * B, -1)

      action = A.reshape(n_guess * B, 1).long()

      qvalues = QS.gather(1, action).squeeze()
      qvalues = qvalues.view(n_guess, B)

      with torch.no_grad():
        QS_t, _ = Q_t(S)
        QS_t = QS_t[1:]
        QS_t = QS_t.view(n_guess * B, self.config.act_n)
        q2values = torch.max(QS, dim=1).values.view(n_guess, B)

      targets = R.reshape(n_guess, B) + self.config.gamma * q2values

      loss = torch.nn.MSELoss()(targets, qvalues) * D.reshape(n_guess, B)

    # back prop
    optim.zero_grad()
    loss.mean().backward()
    optim.step()

    # Update target network
    if episode % self.config.target_network_update_freq == 0:
      self.Qt.load_state_dict(Q.state_dict())

  def update_networks_policy(self, S, A, R, N):
    L = len(R)
    B = self.config.batch_size

    S = torch.stack(S, dim=0)[:-1, :, :] # L x B x state_N
    S = self.env.replay.state_to_onehot(S) # L x B x obs_N
    A = torch.stack(A, dim=0) # L x B
    N = torch.stack(N, dim=0) # L x B

    discounted_rewards = [r.float() for r in R]
    for i in reversed(range(len(R) - 1)):
      discounted_rewards[i] += self.config.gamma * discounted_rewards[i+1]
    returns = torch.stack(discounted_rewards, dim=0) - self.baseline # L x B
    if 'baseline' in self.config.algo:
      if self.baseline == 0:
        self.baseline = returns.mean().item()
      else:
        self.baseline = 0.9 * self.baseline + 0.1 * returns.mean().item()

    for _ in range(self.config.policy_train_iters):
      self.optim.zero_grad()

      logit_probs, _ = self.pi(S)
      log_probs = torch.nn.LogSoftmax(dim=-1)(logit_probs).gather(2, A.unsqueeze(2)).view(L, B)
      n = torch.arange(L).to(self.config.device)
      loss = -((self.config.gamma ** n).unsqueeze(1) * returns * log_probs * N).mean()
      loss.backward()

      self.optim.step()


  # Play episodes
  # Training function
  def train(self):

    # epsilon greedy exploration
    if self.config.algo == 'Q':
      network = self.Q
    elif 'policy' in self.config.algo:
      network = self.pi
    network.epsilon = self.config.epsilon_start

    testRs = []
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(self.config.n_episodes)
    for epi in pbar:

      # Play an episode and log episodic reward
      if self.config.algo == 'Q':
        S, A, R, N = self.env.play_episode_rb(network.policy)
      elif 'policy' in self.config.algo:
        S, A, R, N = self.env.play_episode(network.policy)

      # Train after collecting sufficient experience
      if epi >= self.config.train_after_episodes:

        # Train for TRAIN_EPOCHS
        for tri in range(self.config.train_epochs):
          if self.config.algo == 'Q':
            self.update_networks(epi)
          elif 'policy' in self.config.algo:
            self.update_networks(S, A, R, N)

      # Evaluate for TEST_EPISODES number of episodes
      Rews = []
      for _ in range(self.config.test_epochs):
        S, A, R, _ = self.env.play_episode(network.test_policy)
        Rews += [(sum(R) / len(R)).mean().item()]
      testRs += [sum(Rews)/self.config.test_epochs]

      # Update progress bar
      last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
      pbar.set_description("R25(%g)" % (last25testRs[-1]))

      if self.config.log_path:
        with open(f"{self.config.log_path}_log.txt", "a") as f:
          f.write(f"{epi},{last25testRs[-1]}\n")
        if epi % 25 == 0:
          torch.save(network, f"{self.config.log_path}_DRQN_weights_epi_{epi}.pt")
          torch.save(self.env.replay.states, f"{self.config.log_path}_states_epi_{epi}.pt")
          torch.save(self.env.replay.actions, f"{self.config.log_path}_actions_epi_{epi}.pt")
          torch.save(self.env.replay.rewards, f"{self.config.log_path}_rewards_epi_{epi}.pt")
          torch.save(self.env.replay.not_done, f"{self.config.log_path}_not_done_epi_{epi}.pt")

    pbar.close()
    print("Training finished!")

    return last25testRs

