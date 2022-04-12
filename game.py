import torch
import utils

class Wordle:

  def __init__(self, config):
    self.config = config
    self.device = config.device

    self.N = config.n_letters
    self.batch_size = config.batch_size

    if config.alphabet == 'reduced':
      to_ord = utils.to_ord_reduced
    else:
      to_ord = utils.to_ord

    with open(f"{config.path}_answer.txt", "r") as f:
      answers = f.read().splitlines()

    self.answers = torch.stack([to_ord(a) for a in answers]).to(self.device)

    with open(f"{config.path}_guess.txt", "r") as f:
      guesses = f.read().splitlines()

    self.guesses = torch.stack([to_ord(g) for g in guesses]).to(self.device)
    self.guesses = torch.cat([self.answers, self.guesses])


  def new_game(self):
    self.num_guesses = 0
    self.total_num_guesses = 0
    self.batch_num_guesses = 0
    self.guess_history = torch.empty((self.batch_size, 0, self.N), device=self.device)
    self.cur_answer_idx = torch.randint(len(self.answers), (self.batch_size,), device=self.device)
    self.cur_answer = self.answers[self.cur_answer_idx]
    self.game_not_done = torch.ones(self.batch_size, dtype=int, device=self.device)

  def guess(self, word):
    self.num_guesses += 1

    is_word = utils.batched_in(word, self.guesses)
    green, yello = utils.feedback(word, self.cur_answer)

    already_guessed = torch.zeros((self.batch_size,), device=self.device)
    for b in range(self.batch_size):
      already_guessed[b] = utils.single_in(word[b], self.guess_history[b])

    self.guess_history = torch.cat([self.guess_history, word.unsqueeze(1)], dim=1)

    green *= is_word.view(-1, 1)
    yello *= is_word.view(-1, 1)

    self.total_num_guesses += self.game_not_done.sum().item()
    self.batch_num_guesses += self.game_not_done
    self.game_not_done &= (green == 0).any(dim=1).int()
    # game is not done AND it is a word AND we haven't guessed it yet rewards 3 points for every green and 1 for each yellow
    reward = self.game_not_done * is_word * (1 - already_guessed) * (green.sum(dim=1) * 5 + yello.sum(dim=1))
    # however if we guess a valid guess twice we want to punish it
    reward += self.game_not_done * is_word * already_guessed * (0)
    # penalty of 50 points i we continue the game but don't win
    reward += self.game_not_done * is_word * (0)
    # game is not done AND it is not a word is -1000 (punish non-words)
    reward += self.game_not_done * (1 - is_word) * (-1000)
    reward += (1 - self.game_not_done) * 15

    return reward, green, yello


# human playable version
class WordleHuman:

  def __init__(self, config):
    self.config = config
    self.batch_size = config.batch_size

    self.wordle = Wordle(config)


  def print_history(self, batch):
    history = self.history[batch]
    for guess, green, yello, reward in history:
      for i, ch in enumerate(guess):
        assert green[i] == 0 or yello[i] == 0
        if green[i] > 0:
          utils.print_green(ch, end="")
        elif yello[i] > 0:
          utils.print_yello(ch, end="")
        else:
          print(ch, end="")
      print(f" R={reward}")

  def play(self):
    self.history = [[] for _ in range(self.batch_size)]

    self.wordle.new_game()

    while (self.wordle.game_not_done > 0).any():
      inputs = []
      guesses = []
      for batch in range(self.batch_size):
        if self.wordle.game_not_done[batch]:
          self.print_history(batch)

          inputs.append(input())
          print("")
        else:
          inputs.append(self.history[batch][-1][0])

        guesses.append(utils.to_ord(inputs[-1]))

      guesses = torch.stack(guesses)

      reward, green, yello = self.wordle.guess(guesses)
      for batch in range(self.batch_size):
        self.history[batch].append((inputs[batch], green[batch], yello[batch], reward[batch]))

    for batch in range(self.batch_size):
      self.print_history(batch)
      print("")


