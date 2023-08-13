"""Utility functions for Input/Output."""

import os
import socket
import time
import pickle

ON_CLUSTER = True if "charles" in socket.gethostname() else False


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)
        
def save(data, file):
    """
    Saves data to a file.
    """

    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    with open(file + '.pkl', 'wb') as f:
        pickle.dump(data, f)


def load(file):
    """
    Loads data from file.
    """

    with open(file + '.pkl', 'rb') as f:
        data = pickle.load(f)

    if hasattr(data, 'reset_theano_functions'):
        data.reset_theano_functions()

    return data

def is_on_cluster():
    return True if "charles" in socket.gethostname() else False


def get_timestamp():
    formatted_time = time.strftime("%d_%m_%y_%H_%M_%S")
    return formatted_time


def get_project_root():
    if ON_CLUSTER:
        return "/home/s1638128/deployment/lfi"
    else:
        return os.environ["LFI_PROJECT_DIR"]


def get_log_root():
    if ON_CLUSTER:
        return "/home/s1638128/tmp/lfi/log"
    else:
        return os.path.join(get_project_root(), "log")


def get_data_root():
    if ON_CLUSTER:
        return os.path.join(get_project_root(), "data")
    else:
        return os.path.join(get_project_root(), "data")


def get_output_root():
    if ON_CLUSTER:
        return "/home/s1638128/tmp/lfi/out"
    else:
        return os.path.join(get_project_root(), "out")


def get_checkpoint_root():
    if ON_CLUSTER:
        return "/home/s1638128/tmp/lfi/checkpoint"
    else:
        return os.path.join(get_project_root(), "checkpoint")
