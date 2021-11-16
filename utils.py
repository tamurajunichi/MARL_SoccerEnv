import os
import numpy as np
import hfo_py
from environment import ACTION_LOOKUP


# onpolicy vs offpolicyの論文：https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/DeepRL16-hausknecht.pdf
# on_policy_mc = Σ^T_i=t(γ^(i-t)*r_i)
# mixing_update: y = beta*on_policy_mc + (1-beta)*q_learning
def add_on_policy_mc(trajectory):
    n=len(trajectory)
    n_step_returns = np.zeros((n,))
    n_step_returns[n-1] = trajectory[n-1]["reward"]
    # 最後のステップはrewardの値
    trajectory[n-1]["n_step"] = n_step_returns[n-1]
    exp_n_step_returns = np.zeros((n,))
    exp_n_step_returns[n-1] = trajectory[n-1]["int_reward"]
    trajectory[n-1]["exp_n_step"] = exp_n_step_returns[n-1]
    dis = 0.99
    # range(start, stop, step)
    # 最後のステップから割り引いていく報酬を計算
    for i in range(n-2,-1,-1):
        r = trajectory[i]["reward"]
        target_r = n_step_returns[i+1]
        n_step_returns[i] = r+dis*target_r
        trajectory[i]["n_step"] = n_step_returns[i]

        exp_r = trajectory[i]["int_reward"]
        target_exp_r = exp_n_step_returns[i+1]
        exp_n_step_returns[i] = exp_r+dis*target_exp_r
        trajectory[i]["exp_n_step"] = exp_n_step_returns[i]
    return trajectory


# ActorのNNから得られるアクションを環境で扱えるように変換
def suit_action(action):
    ret_act = np.zeros(6)
    ret_act[0] = np.argmax(action[0:3])
    ret_act[1:6] = action[3:8]
    return ret_act

def random_action():
    max_a = [1, 1, 1, 100, 180, 180, 100, 180]
    min_a = [-1, -1, -1, 0, -180, -180, 0, -180]
    action = np.random.uniform(min_a, max_a)
    return action

def take_action(action, env):
    action_type = ACTION_LOOKUP[action[0]]
    if action_type == hfo_py.DASH:
        env.env.act(action_type, action[1], action[2])
    elif action_type == hfo_py.TURN:
        env.env.act(action_type, action[3])
    elif action_type == hfo_py.KICK:
        env.env.act(action_type, action[4], action[5])

def translation(var):
    s = "("
    if hasattr(var, "__iter__"):
        for v in var:
            s += str(v).replace('.', '_')+"/"
    else:
        s += str(var).replace('.', '_')
    s+=")"
    return s


class Logger:
    def __init__(self,n,log_size):
        self.log_size = log_size
        self.log_array = np.zeros((n, log_size))
        self.n = 0

    def add(self,*data):
        if len(data) != self.log_size:
            self.log_array[self.n] = data[0]
        else:
            self.log_array[self.n] = data
        self.n += 1

    def out(self, log_path, file_name):
        save_name = log_path + file_name
        save_log = self.log_array[:self.n,:]
        try:
            np.save(save_name, save_log)
        except FileNotFoundError:
            try:
                os.mkdir(log_path)
                np.save(save_name, save_log)
            except FileExistsError:
                np.save(save_name, save_log)
