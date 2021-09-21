
import hfo_py

class HalfFieldOffense():
    def __init__(self, port):
        self.port = port
        self.env = hfo_py.HFOEnvironment()
        self.env.connectToServer(server_port=self.port,config_dir=hfo_py.get_config_path())
        self.status = hfo_py.IN_GAME

    def step(self, action):
        self.take_action(action)
        self.status = self.env.step()
        reward = self.get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {'status': STATUS_LOOKUP[self.status]}

    def take_action(self, action):
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo_py.NOOP)

    def get_reward(self):
        if self.status == hfo_py.GOAL:
            return 1
        else:
            return 0

    def reset(self):
        while self.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
            # prevent infinite output when server dies
            if self.status == hfo_py.SERVER_DOWN:
                raise ServerDownException("HFO server down!")
        return self.env.getState()


class ServerDownException(Exception):
    pass


ACTION_LOOKUP = {
    0: hfo_py.DASH,
    1: hfo_py.TURN,
    2: hfo_py.KICK,
    3: hfo_py.TACKLE,
    4: hfo_py.CATCH,
}

STATUS_LOOKUP = {
    hfo_py.IN_GAME: 'IN_GAME',
    hfo_py.SERVER_DOWN: 'SERVER_DOWN',
    hfo_py.GOAL: 'GOAL',
    hfo_py.OUT_OF_BOUNDS: 'OUT_OF_BOUNDS',
    hfo_py.OUT_OF_TIME: 'OUT_OF_TIME',
    hfo_py.CAPTURED_BY_DEFENSE: 'CAPTURED_BY_DEFENSE',
}
