import random
import numpy as np
import torch

def seed(seed):
  random.seed(seed)
  np.random.seed(seed * 2)
  torch.manual_seed(seed * 3)

# reduced letter set
reduced = ['a', 'e', 'i', 'o', 'r', 's', 't', 'm', 'n']
reduce_map = {c: i for i, c in enumerate(reduced)}

def to_onehot(str):
  ret = torch.zeros(26 * len(str))
  for i, c in enumerate(str):
    ret[i * 26 + (ord(c) - ord('a'))] = 1
  return ret

def to_onehot_reduced(str):
  ret = torch.zeros(9 * len(str))
  for i, c in enumerate(str):
    ret[i * 9 + reduce_map[c]] = 1
  return ret

def to_ord(str):
  return torch.tensor([ord(c) - ord('a') for c in str])

def to_ord_reduced(str):
  return torch.tensor([reduce_map[c] for c in str])

def elementwise_in(a, b):
  # batched elementwise a in b
  B, N = b.shape
  assert a.shape == (B, N)
  ret = torch.zeros(B, N, dtype=bool, device=a.device)
  for i in range(N):
    ret |= a == b[:, i].view(-1, 1)
  return ret

def batched_in(a, b):
  # returns (B,)
  B, N = a.shape
  M, _ = b.shape
  ret = torch.zeros(B, device=a.device)

  for i in range(B):
    ret[i] = (a[i] == b).all(dim=1).any().float()

  return ret

def single_in(a, b):
  N, = a.shape
  M, _ = b.shape # M x N

  return (a == b).all(dim=1).any().float()


def elementwise_in_unique(a, b, destructive=False):
  # batched unique elementwise a in b
  # noted test case: [1, 1, 3] in [1, 2, 3] == [True, False, True]
  if not destructive:
    a = a.clone()
  B, N = b.shape
  assert a.shape == (B, N)
  ret = torch.zeros(B, N, dtype=bool, device=a.device)
  for i in range(N):
    a_in_b = a == b[:, i].view(-1, 1)
    first_a_in_b_idx = torch.argmax(a_in_b.float(), dim=1)
    has_a_in_b = a_in_b.any(dim=1)

    batches = torch.arange(B)[has_a_in_b]
    first_a_in_b_idx = first_a_in_b_idx[has_a_in_b]

    ret[batches, first_a_in_b_idx] = True
    a[batches, first_a_in_b_idx] = -1

  return ret



def feedback(guess, target, destructive=False):
  if not destructive:
    guess = guess.clone()
    target = target.clone()
  green = guess == target
  guess[green] = -2
  target[green] = -3
  green = green.float()

  yello = elementwise_in_unique(guess, target, destructive=False).float()

  return green, yello

def feedback_str(guess, target):
  return feedback(to_ord(guess).view(1,-1), to_ord(target).view(1, -1))


green_clr = '\033[92m'
yello_clr = '\033[93m'
end_clr = '\033[0m'

def print_green(str, **kwargs):
  global green_clr, end_clr
  print(f"{green_clr}{str}{end_clr}", **kwargs)


def print_yello(str, **kwargs):
  global yello_clr, end_clr
  print(f"{yello_clr}{str}{end_clr}", **kwargs)



