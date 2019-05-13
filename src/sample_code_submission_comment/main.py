# pylint: disable=wrong-import-order, wrong-import-position, import-error
# pylint: disable=missing-docstring
import base64
from datetime import datetime
import os
from os.path import join
import sys

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

#  os.system("pip3 install cryptography")


def mprint(msg):
    """info"""
    cur_time = datetime.now().strftime('%m-%d %H:%M:%S')
    print(f"INFO  [{cur_time}] {msg}")


if len(sys.argv) == 1:
    # default local
    # ROOT_DIR = os.getcwd()
    # DIRS = {
    #     'input': join(ROOT_DIR, 'sample_data'),
    #     'output': join(ROOT_DIR, 'sample_predictions'),
    #     'program': join(ROOT_DIR, 'ingestion_program'),
    #     'submission': join(ROOT_DIR, 'sample_code_submission')
    # }

    ROOT_DIR = os.path.abspath('/Users/xijunli/Desktop/KDDCup2019/starting_kit_0401/')
    DIRS = {
        'input': join(ROOT_DIR, 'sample_data'),
        'output': join(ROOT_DIR, 'sample_predictions'),
        'program': join(ROOT_DIR, 'ingestion_program'),
        'submission': join(ROOT_DIR, 'sample_code_submission')
    }

elif len(sys.argv) == 5:
    # run in codalab
    DIRS = {
        'input': sys.argv[1],
        'output': sys.argv[2],
        'program': sys.argv[3],
        'submission': sys.argv[4]
    }
elif len(sys.argv) == 6 and sys.argv[1] == 'local':
    # full call in local
    DIRS = {
        'input': sys.argv[2],
        'output': sys.argv[3],
        'program': sys.argv[4],
        'submission': sys.argv[5]
    }
else:
    raise ValueError("Wrong number of arguments")
sys.path.append(DIRS['submission'])


def read_public_key():
    with open(join(DIRS['program'], "public_key.pem"), "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )
    return public_key


def encrypt(msg, public_key):
    msg = msg.encode()
    encrypted = public_key.encrypt(
        msg,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted


def print_info():
    ip_addr = os.environ.get('DOCKER_HOST_KDD')
    if ip_addr is not None:
        ip_msg = base64.b64encode(encrypt(ip_addr, PUBLIC_KEY))
        mprint(f"==== {ip_msg}")


# PUBLIC_KEY = read_public_key()
# print_info()


mprint("Import Model")
from model import Model
import json
import signal
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import math

TYPE_MAP = {
    'time': str,
    'cat': str,
    'multi-cat': str,
    'num': np.float64
}


class TimeoutException(Exception):
    pass


class Timer:
    def __init__(self):
        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.remain)
        start_time = time.time()
        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            remain_time = math.ceil(self.total - self.exec)
            self.remain = remain_time

            mprint(f'{pname} success, time spent so far {self.exec} sec')


def read_train(datapath, info):
    train_data = {}
    for table_name, columns in info['tables'].items():
        mprint(f'Table name: {table_name}')

        table_dtype = {key: TYPE_MAP[val] for key, val in columns.items()}

        if table_name == 'main':
            table_path = join(datapath, 'train', 'main_train.data')
        else:
            table_path = join(datapath, 'train', f'{table_name}.data')

        date_list = [key for key, val in columns.items() if val == 'time']

        train_data[table_name] = pd.read_csv(
            table_path, sep='\t', dtype=table_dtype, parse_dates=date_list,
            date_parser=lambda millisecs: millisecs if np.isnan(
                float(millisecs)) else datetime.fromtimestamp(
                    float(millisecs)/1000))

    # get train label
    train_label = pd.read_csv(
        join(datapath, 'train', 'main_train.solution'))['label']
    return train_data, train_label


def read_info(datapath):
    mprint('Read info')
    with open(join(datapath, 'train', 'info.json'), 'r') as info_fp:
        info = json.load(info_fp)
    mprint(f'Time budget for this task is {info["time_budget"]} sec')
    return info


def read_test(datapath, info):
    # get test data
    main_columns = info['tables']['main']
    table_dtype = {key: TYPE_MAP[val] for key, val in main_columns.items()}

    table_path = join(datapath, 'test', 'main_test.data')

    date_list = [key for key, val in main_columns.items() if val == 'time']

    test_data = pd.read_csv(
        table_path, sep='\t', dtype=table_dtype, parse_dates=date_list,
        date_parser=lambda millisecs: millisecs if np.isnan(
            float(millisecs)) else datetime.fromtimestamp(
                float(millisecs) / 1000))
    return test_data


def write_predict(output_dir, dataname, prediction):
    os.makedirs(output_dir, exist_ok=True)
    prediction.rename('label', inplace=True)
    prediction.to_csv(
        join(output_dir, f'{dataname}.predict'), index=False, header=True)


def main():
    datanames = sorted(os.listdir(DIRS['input']))
    mprint(f'Datanames: {datanames}')
    timer = Timer()
    datanames = datanames[0]
    predictions = {}
    for dataname in datanames:
        mprint(f'Read data: {dataname}')
        datapath = join(DIRS['input'], dataname)
        info = read_info(datapath)
        timer.set(info['time_budget'])
        train_data, train_label = read_train(datapath, info)

        mprint('Initalize model')
        with timer.time_limit('Initialization'):
            cmodel = Model(info)

        mprint('Start fitting')
        with timer.time_limit('Fitting'):
            cmodel.fit(train_data, train_label, timer.remain)

        test_data = read_test(datapath, info)
        # user prediction
        mprint('Start prediction')
        with timer.time_limit('Prediction'):
            predictions[dataname] = cmodel.predict(test_data, timer.remain)

        mprint(f'Done, exec_time={timer.exec}')

    # write
    mprint(f'Write results')
    for dataname in datanames:
        write_predict(DIRS['output'], dataname, predictions[dataname])

    mprint(f'Duration: {timer.duration}')
    with open(join(DIRS['output'], 'duration.txt'), 'w') as out_f:
        out_f.write(str(timer.duration))


if __name__ == '__main__':
    main()
