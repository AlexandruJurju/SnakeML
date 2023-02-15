from typing import List

import pygame

from constants import ViewConsts, BoardConsts


def draw_board(window, board: List, offset_x, offset_y) -> None:
    # use y,x for index in board instead of x,y because of changed logic
    # x is line y is column ; drawing x is column and y is line
    for x in range(len(board)):
        for y in range(len(board)):
            x_position = x * ViewConsts.SQUARE_SIZE + offset_x
            y_position = y * ViewConsts.SQUARE_SIZE + offset_y

            match board[y][x]:
                case BoardConsts.SNAKE_BODY:
                    pygame.draw.rect(window, ViewConsts.COLOR_SNAKE_SEGMENT, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                case BoardConsts.WALL:
                    pygame.draw.rect(window, ViewConsts.COLOR_WHITE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                case BoardConsts.APPLE:
                    pygame.draw.rect(window, ViewConsts.COLOR_APPLE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                case BoardConsts.SNAKE_HEAD:
                    pygame.draw.rect(window, ViewConsts.COLOR_SNAKE_HEAD, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
            # draw lines between squares
            pygame.draw.rect(window, ViewConsts.COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE), width=1)
