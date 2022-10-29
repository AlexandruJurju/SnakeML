import numpy as np


class Model:
    def __init__(self, size: int):
        self.size = size + 2
        self.board = np.empty((self.size, self.size), dtype=object)
        self.__make_board()

    def __make_board(self) -> None:
        for i in range(0, self.size):
            for j in range(0, self.size):
                if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                    self.board[i, j] = "W"
                else:
                    self.board[i, j] = "X"
