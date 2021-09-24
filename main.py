import hfo_py
import torch.multiprocessing as mp
from agent import Agent
from environment import HalfFieldOffense
import utils

import socket
from contextlib import closing
import time
import datetime
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
                 log_dir='log', sync_mode=True, fullstate=True, verbose=False, log_game=False):
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


def train(num_episodes, port, process_number):
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

    # ログ保存用のndarray
    log = np.zeros((num_episodes, 6))


    # エージェントと環境の相互作用部分
    state = env.reset()
    episode = 0
    timestep = 0
    episode_timestep = 0
    update_ratio = 0.1
    ro = 0.003
    while True:
        trajectory = []
        episode_reward = 0
        exp_episode_reward = 0

        if timestep > 1000:
            action = agent.action(state)
        else:
            action = agent.random_action()
        next_state, reward, done, info = env.step(suit_action(action))
        done_bool = float(done)

        predicted_state = agent.predictor.predict(state, action)
        exp_reward = np.linalg.norm(np.concatenate((next_state, np.array([reward]))) - predicted_state)*ro

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
        timestep += 1
        episode_timestep += 1
        if done:
            # モンテカルロアップデートで使うものを以下でtransitionに付け加える
            add_on_policy_mc(trajectory)
            # transitionをすべてreplay bufferへいれる
            for i in trajectory:
                replay_buffer.add(i["state"], i["action"], i["next_state"],
                                  i["reward"], i["exp_reward"], i["n_step"],
                                  i["exp_n_step"], i["done"])
            if timestep >= 1000:
                critic_mean, predictor_loss_mean = np.ndarray([0,0,0]), 0
                for i in range(int(episode_timestep*update_ratio)):
                    critic, predictor_loss = agent.learn(replay_buffer)
                    critic_mean += np.ndarray(critic)
                    predictor_loss_mean += predictor_loss_mean
                critic_mean = critic_mean / len(critic_mean)
                predictor_loss_mean = predictor_loss_mean / len(critic_mean)


                # logの追加
                log[episode][0] = episode_reward
                log[episode][1] = exp_episode_reward
                log[episode][2] = critic_mean[0] # current_q.mean().item()
                log[episode][3] = critic_mean[1] # mixed_q.mean().item()
                log[episode][4] = critic_mean[2] # critic_loss (mse current_q - mixed_q)
                log[episode][5] = predictor_loss_mean


            # エピソード終了のリセット
            state, done = env.reset(), False
            episode_reward = 0
            exp_episode_reward = 0
            transitions = []
            episode += 1
            episode_timestep = 0


        if episode >= num_episodes:
            try:
                np.save('./agent_log/{}_{}.npy'.format(process_number,datetime.datetime.now()), log)
            except FileNotFoundError:
                try:
                    os.mkdir('agent_log')
                    np.save('./agent_log/{}_{}.npy'.format(process_number, datetime.datetime.now()), log)
                except FileExistsError:
                    np.save('./agent_log/{}_{}.npy'.format(process_number, datetime.datetime.now()), log)
            break


def main():
    num_episodes = 100000
    viewer = False

    # HalfFieldOffenseサーバーを起動
    server_process, port = start_server(offense_agents=2)
    if viewer:
        viewer_process = start_viewer(port)
    # train(num_episodes,port)

    # pytorchのマルチプロセス
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    num_processes = 2
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(num_episodes,port,rank))
        p.start()
        processes.append(p)
        time.sleep(1)
    for p in processes:
        p.join()
    if viewer:
        close(viewer_process)
    close(server_process)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()