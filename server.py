import socket
from contextlib import closing
import subprocess
import signal
import time
import os

import hfo_py

def find_free_port():
    """Find a random free port. Does not guarantee that the port will still be free after return.
    Note: HFO takes three consecutive port numbers, this only checks one.

    Source: https://github.com/crowdAI/marLo/blob/master/marlo/utils.py

    :rtype:  `int`
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start(frames_per_trial=100,
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
          " --ball-x-max %f --log-dir %s --message-size %s" \
          % (frames_per_trial,
             offense_agents,
             defense_agents, offense_npcs, defense_npcs, port,
             offense_on_ball, seed, ball_x_min, ball_x_max,
             log_dir,10000)
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