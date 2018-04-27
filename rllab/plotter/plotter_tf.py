import atexit
from queue import Empty, Queue
from rllab.sampler.utils import rollout
import numpy as np
import tensorflow as tf
from threading import Thread

__all__ = ['init_worker', 'init_plot', 'update_plot']

process = None
queue = None


def _worker_start(sess=None):
    env = None
    policy = None
    max_length = None
    try:
        if sess != None:
            with sess.as_default():
                while True:
                    msgs = {}
                    # Only fetch the last message of each type
                    while True:
                        try:
                            msg = queue.get_nowait()
                            msgs[msg[0]] = msg[1:]
                        except Empty:
                            break
                    if 'stop' in msgs:
                        break
                    elif 'update' in msgs:
                        env, policy = msgs['update']
                        # env.start_viewer()
                    elif 'demo' in msgs:
                        param_values, max_length = msgs['demo']
                        policy.set_param_values(param_values)
                        rollout(
                            env,
                            policy,
                            max_path_length=max_length,
                            animated=True,
                            speedup=5)
                    else:
                        if max_length:
                            rollout(
                                env,
                                policy,
                                max_path_length=max_length,
                                animated=True,
                                speedup=5)

    except KeyboardInterrupt:
        pass


def _shutdown_worker():
    if process:
        queue.put(['stop'])
        queue.task_done()
        queue.join()
        process.join()


def init_worker(sess=None):
    global process, queue
    queue = Queue()
    if sess != None:
        process = Thread(target=_worker_start, args=(sess, ))
    process.start()
    atexit.register(_shutdown_worker)


def init_plot(env, policy):
    queue.put(['update', env, policy])


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
