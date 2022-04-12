import torch
import utils
import numpy as np

class DRQN(torch.nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.epsilon = 1.

    self.hidden = config.hidden
    self.device = config.device

    if config.model == "rnn":
      self.rnn = torch.nn.LSTM(
          input_size=self.config.obs_n,
          hidden_size=config.hidden,
      ).to(config.device)

    elif config.model == "attention":
      # one hot position encoding
      self.attention_head = torch.nn.Sequential(
          torch.nn.Linear(config.obs_n + config.n_guesses, config.att_hidden), torch.nn.ReLU(),
          torch.nn.Linear(config.att_hidden, 1), torch.nn.Sigmoid()
      ).to(config.device)
      self.encoder = torch.nn.Sequential(torch.nn.Sequential(
          torch.nn.Linear(config.obs_n + config.n_guesses, config.hidden), torch.nn.ReLU(),
          torch.nn.Linear(config.hidden, config.hidden),
      ).to(config.device))
    else:
      raise
    if config.action_rep == 'onehot':
      self.Q = torch.nn.Sequential(
          torch.nn.Linear(config.hidden, config.hidden), torch.nn.ReLU(),
          torch.nn.Linear(config.hidden, config.hidden), torch.nn.ReLU(),
          torch.nn.Linear(config.hidden, self.config.act_n)
      ).to(config.device)
    elif config.action_rep == 'score':
      self.Q = torch.nn.Sequential(
          torch.nn.Linear(config.hidden, config.hidden), torch.nn.ReLU(),
          torch.nn.Linear(config.hidden, config.hidden), torch.nn.ReLU(),
          torch.nn.Linear(config.hidden, self.config.n_letters * self.config.n_alphabet)
      ).to(config.device)

      if self.config.alphabet == 'reduced':
        to_onehot = utils.to_onehot_reduced
      else:
        to_onehot = utils.to_onehot
      with open(f"{config.path}_answer.txt", "r") as f:
        answers = f.read().splitlines()

      with open(f"{config.path}_guess.txt", "r") as f:
        guesses = f.read().splitlines()

      answers = torch.stack([to_onehot(a) for a in answers])
      guesses = torch.stack([to_onehot(g) for g in guesses])
      self.dictionary = torch.cat([answers, guesses])
      self.words = self.dictionary.T.unsqueeze(0).to(self.device) # 1 x num_letters x num_words


  def forward(self, x, hidden=None):
    if len(x.shape) == 2:
      L, N = x.shape
      B = 1
    else:
      L, B, N = x.shape
    if self.config.model == "rnn":
      if hidden is None:
        hidden = torch.zeros((1, B, self.hidden), device=self.device), torch.zeros((1, B, self.hidden), device=self.device)
      out, hidden_out = self.rnn(x.view(L, B, N), hidden)

    elif self.config.model == "attention":
      if hidden is None:
        hidden, l_0 = torch.zeros((B, self.hidden), device=self.device), 0
      else:
        hidden, l_0 = hidden
      x = x.view(L, B, N)
      out = torch.zeros((L, B, self.hidden), device=self.device)
      for l in range(L):
        position = torch.zeros((B, self.config.n_guesses), device=self.device)
        position[:, l_0] = 1
        init = torch.cat([x[l], position], dim=1)
        attention = self.attention_head(init)
        encoding = self.encoder(init)

        encoding *= attention
        hidden += encoding
        out[l] = hidden
      hidden_out = hidden, l_0 + L
    action = self.Q(out.view(L * B, self.hidden))

    if self.config.action_rep == "score":
      action = action.unsqueeze(2) # L*B x num_letters x 1
      score = (action * self.words).transpose(1,2).sum(dim=2) # L*B x num_words
      action = score

    return action.view(L, B, self.config.act_n), hidden_out

  # Create epsilon-greedy policy
  def policy(self, obs, hidden, test=False):

    obs = obs.view(1, -1, self.config.obs_n).to(self.device)

    if self.config.algo == 'Q':
      # With probability EPSILON, choose a random action
      # Rest of the time, choose argmax_a Q(s, a)
      with torch.no_grad():
        qvalues, hidden_out = self.forward(obs, hidden)

      if self.config.experiment == 'no_knowledge':
        if not test and np.random.rand() < self.epsilon:
          action = torch.randint(self.config.n_alphabet, (self.config.batch_size, self.config.n_letters)).to(self.device)
        else:
          action = torch.argmax(qvalues.view(self.config.batch_size, self.config.n_letters, self.config.n_alphabet), dim=2)
      elif self.config.experiment == 'with_dict':
        if not test and np.random.rand() < self.epsilon:
          action = torch.randint(self.config.act_n, (self.config.batch_size,)).to(self.device)
        else:
          action = torch.argmax(qvalues.view(self.config.batch_size, self.config.act_n), dim=1)

      # Epsilon update rule: Keep reducing a small amount over
      # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
      self.epsilon = max(self.config.epsilon_end, self.epsilon - (1.0 / self.config.steps_max))

      return action, hidden_out
    elif 'policy' in self.config.algo:
      # policy gradient
      with torch.no_grad():
        logit_probs, hidden_out = self.forward(obs, hidden)
        probs = torch.nn.Softmax(dim=-1)(logit_probs)
      return torch.multinomial(probs.squeeze(0), 1).squeeze(1), hidden_out

  def test_policy(self, obs, hidden):
    return self.policy(obs, hidden, test=True)

