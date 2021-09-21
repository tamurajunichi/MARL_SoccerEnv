import hfo_py
import torch.multiprocessing as mp
from agent import Agent
from environment import HalfFieldOffense
import utils

import socket
from contextlib import closing
import time
import subprocess
import os
import signal
import numpy as np

# onpolicy vs offpolicyの論文：https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/DeepRL16-hausknecht.pdf
# on_policy_mc = Σ^T_i=t(γ^(i-t)*r_i)
# mixing_update: y = beta*on_policy_mc + (1-beta)*q_learning
def add_on_policy_mc(trajectory):
    r = 0
    exp_r = 0
    dis = 0.99
    # range(start, stop, step)
    for i in range(len(trajectory)-1,-1,-1):
        r = trajectory[i]["reward"]+dis*r
        trajectory[i]["n_step"] = r
        exp_r = trajectory[i]["exp_reward"]+dis*exp_r
        trajectory[i]["exp_n_step"] = exp_r


# ActorのNNから得られるアクションを環境で扱えるように変換
def suit_action(action):
    ret_act = np.zeros(6)
    ret_act[0] = np.argmax(action[0:3])
    ret_act[1:6] = action[3:8]
    return ret_act


def find_free_port():
    """Find a random free port. Does not guarantee that the port will still be free after return.
    Note: HFO takes three consecutive port numbers, this only checks one.

    Source: https://github.com/crowdAI/marLo/blob/master/marlo/utils.py

    :rtype:  `int`
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_server(frames_per_trial=100,
                 offense_agents=1, defense_agents=0, offense_npcs=0, defense_npcs=0,
                 offense_on_ball=0,
                 seed=-1,
                 ball_x_min=0.0, ball_x_max=0.2,
                 log_dir='log', sync_mode=True, fullstate=True, verbose=False, log_game=True):
    hfo_path = hfo_py.get_hfo_path()
    port = find_free_port()
    cmd = hfo_path + \
          " --headless --frames-per-trial %i --offense-agents %i" \
          " --defense-agents %i --offense-npcs %i --defense-npcs %i" \
          " --port %i --offense-on-ball %i --seed %i --ball-x-min %f" \
          " --ball-x-max %f --log-dir %s" \
          % (frames_per_trial,
             offense_agents,
             defense_agents, offense_npcs, defense_npcs, port,
             offense_on_ball, seed, ball_x_min, ball_x_max,
             log_dir)
    if not sync_mode: cmd += " --no-sync"
    if fullstate:     cmd += " --fullstate"
    if verbose:       cmd += " --verbose"
    if not log_game:  cmd += " --no-logging"
    print('Starting server with command: %s' % cmd)
    server_process = subprocess.Popen(cmd.split(' '), shell=False)
    time.sleep(10)  # Wait for server to startup before connecting a player
    return server_process, port


def start_viewer(port):
    """
    Starts the SoccerWindow visualizer. Note the viewer may also be
    used with a *.rcg logfile to replay a game. See details at
    https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
    """
    cmd = hfo_py.get_viewer_path() + \
          " --connect --port %d" % (port)
    viewer = subprocess.Popen(cmd.split(' '), shell=False)
    return viewer


def close(server_process):
    if server_process is not None:
        try:
            os.kill(server_process.pid, signal.SIGKILL)
        except Exception:
            pass


def logging(trajectory):
    pass


def train(port):
    # アクションの最大と最小
    max_a = [1, 1, 1, 100, 180, 180, 100, 180]
    min_a = [-1, -1, -1, 0, -180, -180, 0, -180]
    state_dim = 68
    action_dim = len(max_a)

    # 環境、エージェント、メモリの生成　メモリはログ出力のためメイン処理ないで記述し、エージェントに渡す。
    env = HalfFieldOffense(port)
    agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_a, min_action=min_a)

    # Experience Replayで使用するリプレイバッファのインスタンス化
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # エージェントと環境の相互作用部分
    state = env.reset()
    episodes = 0
    while True:
        trajectory = []
        episode_reward = 0
        exp_episode_reward = 0

        action = agent.action(state)
        next_state, reward, done, info = env.step(suit_action(action))
        done_bool = float(done)

        predicted_state = agent.predictor.predict(state, action)
        exp_reward = np.linalg.norm(np.concatenate((next_state, np.array([reward]))) - predicted_state)

        trajectory.append({"state": state,
                            "action": action,
                            "next_state": next_state,
                            "reward": reward,
                            "exp_reward": exp_reward,
                            "done": done_bool
                            })

        state = next_state
        episode_reward += reward
        exp_episode_reward += exp_reward
        state = next_state
        if done:
            # モンテカルロアップデートで使うものを以下でtransitionに付け加える
            add_on_policy_mc(trajectory)
            # transitionをすべてreplay bufferへいれる
            for i in trajectory:
                replay_buffer.add(i["state"], i["action"], i["next_state"],
                                  i["reward"], i["exp_reward"], i["n_step"],
                                  i["exp_n_step"], i["done"])
            agent.learn(replay_buffer)


            # エピソード終了のリセット
            state, done = env.reset(), False
            episode_reward = 0
            exp_episode_reward = 0
            transitions = []
            episode_timesteps = 0
            episodes += 1
            print(episodes)


def main():
    # HalfFieldOffenseサーバーを起動
    server_process, port = start_server(offense_agents=2)
    viewer_process = start_viewer(port)
    # train(num_episodes,port)

    # pytorchのマルチプロセス
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    num_processes = 2
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(port,))
        p.start()
        processes.append(p)
        time.sleep(3)
    for p in processes:
        p.join()
    close(viewer_process)
    close(server_process)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()