import utils
import torch

assert (utils.elementwise_in(
    torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [1, 3, 5],
                  [1, 2, 3]]),
    torch.tensor([[1, 3, 5],
                  [1, 3, 5],
                  [1, 3, 5],
                  [1, 3, 5],
                  [2, 3, 4]])
  ) == torch.tensor([[1, 0, 1],
                     [0, 1, 0],
                     [0, 0, 0],
                     [1, 1, 1],
                     [0, 1, 1]]).bool()).all()
assert (utils.elementwise_in_unique(
    torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [1, 3, 5],
                  [1, 2, 3],
                  [1, 1, 3],
                  [1, 1, 3],
                  [1, 1, 1]]),
    torch.tensor([[1, 3, 5],
                  [1, 3, 5],
                  [1, 3, 5],
                  [1, 3, 5],
                  [2, 3, 4],
                  [1, 2, 3],
                  [4, 1, 1],
                  [1, 2, 1]])
  ) == torch.tensor([[1, 0, 1],
                     [0, 1, 0],
                     [0, 0, 0],
                     [1, 1, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0],
                     [1, 1, 0]]).bool()).all()


print("All tests passed")

# 1 1 3

# 1 3 5
