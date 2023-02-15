class BaseState:
    def __init__(self, name: str, target_state_name: str, state_manager):
        self.name = name
        self.target_state_name = target_state_name
        self.state_manager = state_manager

        self.transition = False

        self.data_to_send = {}
        self.data_received = {}

    def trigger_transition(self):
        self.transition = True

    def set_target_state_name(self, target: str):
        self.target_state_name = target

    def start(self):
        pass

    def run(self, surface, time_delta):
        pass

    def end(self):
        pass
