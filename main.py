import hfo_py
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from environment import HalfFieldOffense
import server
import memory
import utils

import time
import datetime
import numpy as np


def train(num_episodes, port, process_number, exploration, ro, seed, td3):
    # アクションの最大と最小
    max_a = [1, 1, 1, 100, 180, 180, 100, 180]
    min_a = [-1, -1, -1, 0, -180, -180, 0, -180]
    state_dim = 59
    action_dim = len(max_a)

    # 環境、エージェント、メモリの生成　メモリはログ出力のためメイン処理ないで記述し、エージェントに渡す。
    env = HalfFieldOffense(port)
    agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_a, min_action=min_a, exploration=exploration, max_episode=num_episodes, td3=td3)

    # Experience Replayで使用するリプレイバッファのインスタンス化
    replay_buffer = memory.ReplayBuffer(state_dim, action_dim)

    # ロガーの設定
    max_timestep = 100000000
    episode_logger = utils.Logger(num_episodes,5)
    timestep_logger = utils.Logger(max_timestep, 59)
    board_writer = SummaryWriter(log_dir="./logs")

    # エピソード初めの最初の状態s0
    state = env.reset()

    # if you add a new feature in lowlevel_feature_extractor.cpp
    # sep_idx = 7
    # position, state = state[1:sep_idx], np.append(state[0:1],state[sep_idx:])

    episode = 0
    timestep = 0
    episode_timestep = 0
    update_ratio = 0.1
    kick_count = 0
    kickable = False
    episode_reward = 0
    int_episode_reward = 0
    trajectory = []
    try:
        while True:
            # timestepごとのログ保存
            #timestep_logger.add(np.append(state,position))
            timestep_logger.add(state)

            # actionの選択
            if timestep > 1000:
                action = agent.action(state)
            else:
                action = agent.random_action()
            s_action = utils.suit_action(action)

            # 現在の状態stateがキック可能で、actionにキックが選ばれていた場合
            #kick_reward = 0
            #if kickable != -1.0 and s_action[0] == 2:
            #    kick_count += 1
                # 0.01 * キックパワー
                #kick_reward = 0.01*s_action[4]/kick_count

            # 環境のstepを実行
            next_state, reward, done, info = env.step(s_action)
            #reward += kick_reward

            # if you add a new feature in lowlevel_feature_extractor.cpp
            # position, next_state = next_state[1:sep_idx], np.append(next_state[0:1],next_state[sep_idx:])
            # kickable = position[-1]

            done_bool = float(done)

            # explorationによってpredictから返ってくる変数の数が変わります
            if exploration == "EG":
                int_reward = 0
            elif exploration == "CE" or exploration == "CE+EG":
                predict = agent.predict(state, action)
                int_reward = np.linalg.norm(np.concatenate((next_state, np.array([reward]))) - predict) * ro
            elif exploration == "RND" or exploration == "RND+EG":
                predict,target = agent.predict(next_state, action)
                int_reward = np.linalg.norm(target - predict) * ro
            else:
                raise(ValueError)

            trajectory.append({"state": state,
                                "action": action,
                                "next_state": next_state,
                                "reward": reward,
                                "int_reward": int_reward,
                                "done": done_bool
                                })

            episode_reward += reward
            int_episode_reward += int_reward
            state = next_state
            timestep += 1
            episode_timestep += 1
            if done:
                # モンテカルロアップデートで使うものを以下でtransitionに付け加える
                trajectory = utils.add_on_policy_mc(trajectory)
                # trajectoryをすべてreplay bufferへいれる
                for i in trajectory:
                    replay_buffer.add(i["state"], i["action"], i["next_state"],
                                      i["reward"], i["int_reward"], i["n_step"],
                                      i["int_n_step"], i["done"])
                if timestep >= 1000:
                    critic_mean, predictor_loss_mean = 0, 0
                    for i in range(int(episode_timestep*update_ratio)):
                        critic, predictor_loss = agent.learn(replay_buffer)
                        critic_mean += critic
                        predictor_loss_mean += predictor_loss_mean
                    try:
                        critic_mean = critic_mean / int(episode_timestep*update_ratio)
                        predictor_loss_mean = predictor_loss_mean / int(episode_timestep*update_ratio)
                    except ZeroDivisionError:
                        critic_mean = 0
                        predictor_loss_mean = 0

                    # episodeロガー
                    episode_logger.add(episode_reward, int_episode_reward, critic_mean, predictor_loss_mean, kick_count)
                    board_writer.add_scalar("episode_reward/episodes", episode_reward, episode)
                    board_writer.add_scalar("int_episode_reward/episodes", int_episode_reward, episode)
                    board_writer.add_scalar("mixed_q/episodes", critic_mean, episode)
                    board_writer.add_scalar("predictor_loss/episodes", predictor_loss_mean, episode)



                # エピソード終了のリセット
                agent.end_episode()
                state, done = env.reset(), False
                # position, state = state[1:sep_idx], np.append(state[0:1], state[sep_idx:])
                episode_reward = 0
                exp_episode_reward = 0
                trajectory = []
                episode += 1
                episode_timestep = 0
                kick_count = 0

            # 全エピソード終了後の処理
            if episode >= num_episodes:
                out_filename = '{}_{}_{}.npy'.format(datetime.datetime.now(),seed,exploration+"-ro:"+str(ro))
                episode_logger.out(log_path='./agent_log/',file_name=out_filename)
                timestep_logger.out(log_path='./agent_log/',file_name="timestep-"+out_filename)
                break
    except KeyboardInterrupt:
        out_filename = '{}_{}_{}.npy'.format(datetime.datetime.now(), seed, exploration + "-ro:" + str(ro))
        episode_logger.out(log_path='./agent_log/', file_name=out_filename)
        timestep_logger.out(log_path='./agent_log/', file_name="timestep-" + out_filename)


def main():
    # trainに渡す設定
    num_episodes = 50000
    exploration = ["RND", "CE","RND+EG","CE+EG","EG"]
    ro = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    viewer = False
    td3 = True

    for i in range(5):
        seed = np.random.randint(0, 1000000000)
        # pytorchのマルチプロセス
        # https://pytorch.org/docs/stable/notes/multiprocessing.html
        num_processes = 1
        # HalfFieldOffenseサーバーを起動
        server_process, port = server.start(offense_agents=num_processes)
        if viewer:
            viewer_process = server.start_viewer(port)
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(num_episodes,port,rank,exploration[0],ro[3],seed,td3))
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
