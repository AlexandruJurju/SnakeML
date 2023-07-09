# SnakeML

## Dependencies
Dependencies required to run the program:

1. numpy
2. pygame
3. pygame-gui
4. python3.10

To install the dependencies, run `pip3 install -r requirements.txt`


## Project Description
The Snake game is a classic arcade-style game where a player controls a snake that moves around a grid, aiming to eat food and grow longer. However, unlike traditional snake games, the movement of the snake is controlled by a neural network.
The neural network predicts the next move of the snake using as input a reduced vision of the game board. This reduced vision is implemented using "vision lines", lines that start at the head of the snake and go in different directions until they hit a wall. 
The vision lines can be drawn either using 4 directions (UP, DOWN, LEFT, RIGHT) or 8 directions (UP, DOWN, LEFT, RIGHT + diagonal directions)
Each vision line stores 3 values: distance from head to wall, distance from head to apple and distance from head to the first body segment of the snake.
The neural network can be trained using 2 different methods:
1. Genetic algorithms
2. Backpropagation algorithm


## Getting started

The project folder containes the executable of the project.


