import hfo_py
import torch.multiprocessing as mp
from agent import Agent
from environment import HalfFieldOffense
import server
import memory
import utils

import time
import datetime
import numpy as np


def train(num_episodes, port, process_number, exploration, ro, seed):
    # アクションの最大と最小
    max_a = [1, 1, 1, 100, 180, 180, 100, 180]
    min_a = [-1, -1, -1, 0, -180, -180, 0, -180]
    state_dim = 59
    action_dim = len(max_a)

    # 環境、エージェント、メモリの生成　メモリはログ出力のためメイン処理ないで記述し、エージェントに渡す。
    env = HalfFieldOffense(port)
    agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_a, min_action=min_a, exploration=exploration)

    # Experience Replayで使用するリプレイバッファのインスタンス化
    replay_buffer = memory.ReplayBuffer(state_dim, action_dim)

    # ログ保存用
    episode_logger = utils.Logger(num_episodes,6)
    timestep_logger = utils.Logger(100000000, 64)

    # エージェントと環境の相互作用部分
    state = env.reset()

    # if you add a new feature in lowlevel_feature_extractor.cpp
    sep_idx = 7
    position, state = state[1:sep_idx], np.append(state[0:1],state[sep_idx:])

    episode = 0
    timestep = 0
    episode_timestep = 0
    update_ratio = 0.1
    kick_count = 0
    kickable = False
    while True:
        trajectory = []
        episode_reward = 0
        exp_episode_reward = 0
        # timestepごとのログ保存
        timestep_logger.add(np.append(state,position))

        # actionの選択
        if timestep > 1000:
            action = agent.action(state)
        else:
            action = agent.random_action()
        action = utils.suit_action(action)

        # 現在の状態stateがキック可能で、actionにキックが選ばれていた場合
        if kickable and action[0] == 2:
            kick_count += 1

        # 環境のstepを実行
        next_state, reward, done, info = env.step(action)

        # if you add a new feature in lowlevel_feature_extractor.cpp
        position, next_state = next_state[1:sep_idx], np.append(next_state[0:1],next_state[sep_idx:])
        kickable = position

        done_bool = float(done)

        if exploration == "EG":
            exp_reward = 0
        elif exploration == "CE" or exploration == "CE+EG":
            predicted_state = agent.predictor.predict(state, action)
            exp_reward = np.linalg.norm(np.concatenate((next_state, np.array([reward]))) - predicted_state) * ro
        elif exploration == "RND" or exploration == "RND+EG":
            predicted_state = agent.rnd_predictor.predict(state)
            exp_reward = np.linalg.norm(next_state - predicted_state) * ro
        else:
            raise(ValueError)

        trajectory.append({"state": state,
                            "action": action,
                            "next_state": next_state,
                            "reward": reward,
                            "exp_reward": exp_reward,
                            "done": done_bool
                            })

        episode_reward += reward
        exp_episode_reward += exp_reward
        state = next_state
        timestep += 1
        episode_timestep += 1
        if done:
            # モンテカルロアップデートで使うものを以下でtransitionに付け加える
            trajectory = utils.add_on_policy_mc(trajectory)
            # transitionをすべてreplay bufferへいれる
            for i in trajectory:
                replay_buffer.add(i["state"], i["action"], i["next_state"],
                                  i["reward"], i["exp_reward"], i["n_step"],
                                  i["exp_n_step"], i["done"])
            if timestep >= 1000:
                critic_mean, predictor_loss_mean = np.array([0,0,0],dtype='float64'), 0
                for i in range(int(episode_timestep*update_ratio)):
                    critic, predictor_loss = agent.learn(replay_buffer)
                    critic_mean += np.array(critic)
                    predictor_loss_mean += predictor_loss_mean
                critic_mean = critic_mean / len(critic_mean)
                predictor_loss_mean = predictor_loss_mean / len(critic_mean)

                # episodeロガー
                episode_logger.add(episode_reward, exp_episode_reward, critic_mean[0], critic_mean[1], critic_mean[2], predictor_loss_mean, kick_count)

            # エピソード終了のリセット
            state, done = env.reset(), False
            position, state = state[1:sep_idx], np.append(state[0:1], state[sep_idx:])
            episode_reward = 0
            exp_episode_reward = 0
            transitions = []
            episode += 1
            episode_timestep = 0
            kick_count = 0

        # 全エピソード終了後の処理
        if episode >= num_episodes:
            out_filename = '{}_{}_{}.npy'.format(datetime.datetime.now(),seed,exploration+"-ro:"+str(ro))
            episode_logger.out(log_path='./agent_log/',file_name=out_filename)
            timestep_logger.out(log_path='./agent_log/',file_name="timestep-"+out_filename)
            break


def main():
    # trainに渡す設定
    num_episodes = 10000
    seed = np.random.randint(0, 1000000000)
    exploration = ["RND", "CE","RND+EG","CE+EG"]
    ro = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    viewer = False

    # pytorchのマルチプロセス
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    num_processes = 1
    # HalfFieldOffenseサーバーを起動
    server_process, port = server.start(offense_agents=num_processes)
    if viewer:
        viewer_process = server.start_viewer(port)
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(num_episodes,port,rank,exploration[0],ro[2],seed))
        p.start()
        processes.append(p)
        time.sleep(1)
    for p in processes:
        p.join()
    if viewer:
        server.close(viewer_process)
    server.close(server_process)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()