from paddpg import PADDPG
import predictor
import random
import numpy as np


class Agent(object):
    def __init__(self,state_dim,action_dim,max_action,min_action,agents,discount=0.99,tau=1e-4,exploration="EG"):

        self.min_action = min_action
        self.max_action = max_action
        self.ddpg = PADDPG(state_dim, action_dim, max_action, min_action,agents,exploration)
        self.exploration = exploration
        # predictor別の設定
        if self.exploration == "RND" or self.exploration == "RND+EG":
            self.rnd_predictor = predictor.RND_Predictor(state_dim)
            self.rnd_target = predictor.RND_Target(state_dim)
        elif self.exploration == "CE" or self.exploration == "CE+EG":
            self.predictor = predictor.Predictor(state_dim,action_dim)
        else:
            pass
        self.counter = 0

    def learn(self, replay_buffer, batch_size=64):
        if self.exploration == "RND" or self.exploration == "RND+EG":
                return (self.ddpg.train(replay_buffer, batch_size),
                    self.rnd_predictor.train(replay_buffer, self.rnd_target, batch_size))
        elif self.exploration == "CE" or self.exploration == "CE+EG":
            return (self.ddpg.train(replay_buffer, batch_size),  self.predictor.train(replay_buffer, batch_size))
        else:
            return self.ddpg.train(replay_buffer, batch_size), 0

    def action(self, state):
        if self.exploration == "EG" or self.exploration == "RND+EG" or self.exploration == "CE+EG":
            self.counter += 1
            eps= random.random()
            dec = min(max(0.1,1.0 - float(self.counter)*0.0000002),1)
            if eps < dec:
                action = np.random.uniform(self.min_action, self.max_action)
            else:
                action = self.ddpg.select_action(state)
        else:
            action = self.ddpg.select_action(state)
        return action

    def random_action(self):
        action = np.random.uniform(self.min_action, self.max_action)
        return action

    def predict(self, state, action):
        if self.exploration == "RND" or self.exploration == "RND+EG":
            return self.rnd_predictor.predict(state), self.rnd_target.predict(state)
        elif self.exploration == "CE" or self.exploration == "CE+EG":
            return self.predictor.predict(state, action)
        else:
            pass