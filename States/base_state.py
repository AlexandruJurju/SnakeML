from settings import State


class BaseState:
    def __init__(self, state_id: State, state_manager):
        self.state_id = state_id
        self.target_state = None
        self.state_manager = state_manager

        self.transition = False

        self.data_to_send = {}
        self.data_received = {}

    def trigger_transition(self):
        self.transition = True

    def set_target_state_name(self, target: State):
        self.target_state = target

    def start(self):
        pass

    def run(self, surface, time_delta):
        pass

    def end(self):
        pass
