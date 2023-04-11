import copy
from typing import Dict

import pygame

from States.base_state import BaseState
from game_config import State, ViewSettings


class StateManager:
    def __init__(self):
        self.states: Dict[State, BaseState] = {}
        self.active_state = None
        self.clock = pygame.time.Clock()

    def add_state(self, state: BaseState) -> None:
        if state.state_id not in self.states:
            self.states[state.state_id] = state

    def execute_state(self, surface):
        if self.active_state is not self.states[State.GENETIC_TRAIN_NEW_NETWORK] or ViewSettings.DRAW:
            time_delta = self.clock.tick(ViewSettings.MAX_FPS) / 1000.0
        else:
            time_delta = 0

        if self.active_state is not None:
            self.active_state.run(surface, time_delta)

            if self.active_state.transition:
                if self.active_state.target_state == State.QUIT:
                    return False

                self.active_state.transition = False
                new_state_id = self.active_state.target_state
                self.active_state.end()

                data_to_send_copy = copy.deepcopy(self.active_state.data_to_send)
                self.active_state = self.states[new_state_id]
                self.active_state.data_received = data_to_send_copy
                self.active_state.start()

            if ViewSettings.DRAW:
                pygame.display.flip()

        return True

    def set_initial_state(self, state_id: State):
        if state_id in self.states:
            self.active_state = self.states[state_id]
            self.active_state.start()
