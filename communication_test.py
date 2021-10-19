import server
from environment import HalfFieldOffense
import utils
import time
import torch.multiprocessing as mp


def run(port,process_number):
    env = HalfFieldOffense(port=port)
    state = env.reset()
    episode = 0
    episode_timestep = 0
    while True:
        msg = env.env.hear()
        if msg:
            print('agent{} heard timestep{}, msg:{}'.format(process_number,episode_timestep,msg))

        env.env.say("agent{} {}".format(process_number,episode_timestep))
        action = utils.random_action()
        next_state, reward, done, info = env.step(utils.suit_action(action))
        state = next_state
        episode_timestep += 1
        if done:
            episode_timestep = 0
            episode += 1

            if episode > 10000:
                break


def main():
    num_processes = 3
    process, port = server.start_server(offense_agents=num_processes)
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=run, args=(port,rank))
        p.start()
        processes.append(p)
        time.sleep(1)
    for p in processes:
        p.join()
    server.close(process)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()