import copy
from typing import Dict

from States.base_state import BaseState


class StateManager:
    def __init__(self):
        self.states: Dict[str, BaseState] = {}
        self.active_state: BaseState = None

    def add_state(self, state: BaseState) -> None:
        if state.name not in self.states:
            self.states[state.name] = state

    def run(self, surface, time_delta):
        if self.active_state is not None:
            self.active_state.run(surface, time_delta)

            if self.active_state.transition:
                self.active_state.transition = False
                new_state_name = self.active_state.target_state_name
                self.active_state.end()
                data_to_send_copy = copy.deepcopy(self.active_state.data_to_send)
                self.active_state = self.states[new_state_name]
                self.active_state.data_received = data_to_send_copy
                self.active_state.start()

            if self.active_state.name == "quit":
                return False

        return True

    def set_initial_state(self, name: str):
        if name in self.states:
            self.active_state = self.states[name]
            self.active_state.start()
