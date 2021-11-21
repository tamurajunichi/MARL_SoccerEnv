import hfo_py
import math


class HalfFieldOffense():
    def __init__(self, port):
        self.port = port
        self.env = hfo_py.HFOEnvironment()
        self.env.connectToServer(server_port=self.port,config_dir=hfo_py.get_config_path())
        self.status = hfo_py.IN_GAME

        self.first_step = True
        self.unum = self.env.getUnum()  # uniform number (identifier) of our lone agent
        self.got_kickable_reward = False

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
        current_state = self.env.getState()
        #print("State =",current_state)
        #print("len State =",len(current_state))
        ball_proximity = current_state[53]
        goal_proximity = current_state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity
        kickable = current_state[12]
        ball_ang_sin_rad = current_state[51]
        ball_ang_cos_rad = current_state[52]
        ball_ang_rad = math.acos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = current_state[13]
        goal_ang_cos_rad = current_state[14]
        goal_ang_rad = math.acos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
        ball_dist_goal = math.sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
                                   2.*ball_dist*goal_dist*math.cos(alpha))
        # Compute the difference in ball proximity from the last step
        if not self.first_step:
            ball_prox_delta = ball_proximity - self.old_ball_prox
            kickable_delta = kickable - self.old_kickable
            ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal
        self.old_ball_prox = ball_proximity
        self.old_kickable = kickable
        self.old_ball_dist_goal = ball_dist_goal
        reward = 0
        if not self.first_step:
            mtb = self.__move_to_ball_reward(kickable_delta, ball_prox_delta)
            ktg = 3. * self.__kick_to_goal_reward(ball_dist_goal_delta)
            eot = self.__EOT_reward()
            reward = mtb + ktg + eot
            # print("mtb: %.06f ktg: %.06f eot: %.06f"%(mtb,ktg,eot))

        self.first_step = False
        # print("r =",reward)
        return reward

    def __move_to_ball_reward(self, kickable_delta, ball_prox_delta):
        reward = 0.
        if self.env.playerOnBall().unum < 0 or self.env.playerOnBall().unum == self.unum:
            reward += ball_prox_delta
        if kickable_delta >= 1 and not self.got_kickable_reward:
            print("KICK!")
            reward += 1.
            self.got_kickable_reward = True
        return reward

    def __kick_to_goal_reward(self, ball_dist_goal_delta):
        if (self.env.playerOnBall().unum == self.unum):
            return -ball_dist_goal_delta
        elif self.got_kickable_reward == True:
            return 0.2 * -ball_dist_goal_delta
        return 0.

    def __EOT_reward(self):
        if self.status == hfo_py.GOAL:
            return 5.
        # elif self.status == hfo_py.CAPTURED_BY_DEFENSE:
        #    return -1.
        return 0.

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
