import random
import threading
import time
import traceback
from bisect import bisect
from collections import deque

import numpy as np
from tqdm import tqdm


def parallel_map(
    f,
    iter_,
    num_workers=4,
    unpack=False,
    use_tqdm=True,
    tqdm_leave=True,
    shuffle=False,
    disable=False,
):
    if disable:
        it = tqdm(iter_, dynamic_ncols=True, leave=tqdm_leave, disable=not use_tqdm)
        if unpack:
            return [f(*i) for i in it]
        return list(map(f, it))
    ret = []
    alive = []
    err = []

    def _job(_index, _args):
        try:
            if unpack:
                ret.append([_index, f(*_args)])
            else:
                ret.append([_index, f(_args)])
        except:
            err.append(traceback.format_exc())

    iter_ = list(enumerate(iter_))
    if shuffle:
        random.shuffle(iter_)
    with tqdm(
        total=len(iter_),
        dynamic_ncols=True,
        leave=tqdm_leave,
        disable=not use_tqdm,
        miniters=0,
    ) as bar:
        try:
            for i in iter_:
                t = threading.Thread(target=_job, args=i)
                t.start()
                alive.append(t)
                time.sleep(0.01)
                c = len(alive)
                while len(alive) >= num_workers:
                    alive = [i for i in alive if i.is_alive() or i.join()]
                    bar.update(0)
                    assert not err
                    time.sleep(0.1)
                bar.update(c - len(alive))
            while alive:
                c = len(alive)
                alive = [i for i in alive if i.is_alive() or i.join()]
                bar.update(c - len(alive))
                assert not err
                time.sleep(0.1)
        except AssertionError:
            print(err[0])
            exit()

    ret.sort(key=lambda x: x[0])
    return [i for _, i in ret]


def wheel(a, b):
    assert 1 <= b <= a
    arr = list(range(a))
    while True:
        random.shuffle(arr)
        for i in range(0, a - b + 1, b):
            yield arr[i:i + b]


class PER():
    def __init__(self, size):
        self.queue = deque(maxlen=size)

    def add(self, n):
        self.queue.extend([None] * n)

    def sample(self, n, alpha=0.5, use_rank=True):
        m = [i for i in self.queue if i is not None]
        if m:
            m = max(m)
            p = [m if i is None else i for i in self.queue]
            if use_rank:
                p = 1 / (len(self.queue) - np.argsort(p))
            else:
                p = np.array(p)
            p = np.cumsum(p**alpha)
            p = (p / p[-1]).tolist()
            return [bisect(p, random.random()) for _ in range(n)]
        else:
            return random.sample(range(len(self.queue)), n)

    def update(self, i, p):
        if isinstance(i, int):
            self.queue[i] = p
        else:
            for a, b in zip(i, p):
                self.queue[a] = b
