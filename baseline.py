import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import time
from glob import glob

import numpy as np
from tqdm import tqdm

from solver import Solver
from utils import parallel_map

parser = argparse.ArgumentParser()

parser.add_argument('-n', type=int, default=100)
parser.add_argument('--method', type=str, default='bb', help='method')
parser.add_argument('-s', type=int, default=0, help='start')
parser.add_argument('-e', type=int, default=100, help='end')
parser.add_argument('--step', type=int, default=50, help='number of steps')
parser.add_argument('--step-size', type=int, default=2, help='step size')
parser.add_argument('--temp', type=float, default=10, help='temperature')
parser.add_argument('--parallel', type=int, default=1, help='parallel')

args = parser.parse_args()

if args.method in 'near kmeans gmm'.split():
    ...
elif args.method.startswith('sa'):
    ...
elif args.method.startswith('bb'):
    ...
elif args.method.startswith('bo'):
    ...
else:
    raise NotImplementedError

N = args.n
problems = pickle.load(open(f'data/{N}_10k.pkl', 'rb'))
path = f'exp/{args.method}_{N}_10k'
if os.path.exists(path + '/tmp'):
    shutil.rmtree(path + '/tmp')
os.makedirs(path + '/tmp', exist_ok=True)
os.makedirs(path + '/json', exist_ok=True)


def Nan(x):
    if np.isnan(x):
        return 1e6
    return x


def sa_optimize(func, init_x, init_y, init_t=1, range_x=4, n_iter=50, step_action=12, seed=43, use_tqdm=False):
    random.seed(seed)
    ind = list(range(len(init_x)))
    for t in tqdm(range(n_iter), disable=not use_tqdm, dynamic_ncols=True):
        new_x = init_x.copy()
        random.shuffle(ind)
        new_x[ind[:step_action]] = (new_x[ind[:step_action]] + np.random.randint(low=0, high=range_x, size=step_action)) % range_x
        new_y = func(new_x)
        if init_y - new_y > np.log(random.random()) * (1 - (t + 1) / n_iter) * init_t:
            init_x = new_x
            init_y = new_y
    return init_x, init_y


def worker(param):
    i, p = param
    if os.path.exists(f'{path}/json/{i}.json'):
        return
    if args.method in 'near kmeans gmm'.split():
        p = subprocess.Popen(
            f'python baseline_rule.py --method {args.method} -i {i} -n {args.n}',
            shell=True
        )
        p.communicate()
    elif args.method.startswith('bo'):
        p = subprocess.Popen(
            f'python baseline_hebo.py --method {args.method} -i {i} -n {args.n} --step {args.step}',
            shell=True)
        p.communicate()
    elif args.method.startswith('bb'):
        p = subprocess.Popen(
            f'python baseline_bb.py --method {args.method} -i {i} -n {args.n} --step {args.step}',
            shell=True)
        p.communicate()
    else:
        solver = Solver(p)
        init = np.hstack(solver.assignment)
        ind = np.cumsum([0] + [len(i) for i in solver.assignment])
        ind = [slice(i, j) for i, j in zip(ind, ind[1:])]

        def obj(x):
            x = x.astype(int)
            solver.assignment = [x[i] for i in ind]
            solver.eval_all()
            pl, cl, pu, cu = sum(solver.state.pl), sum(solver.state.cl), solver.state.pu, solver.state.cu
            with open(f'{path}/tmp/{i}_{time.time()-_t:.2f}_{pl}_{cl}_{pu}_{cu}', 'w') as f:
                f.write(f'{x.tolist()}\n{solver.state.sl}\n{solver.state.su}')
            ret = (pl + Nan(pu)) * 1e6 + Nan(cl) + Nan(cu)
            return ret

        _t = time.time()
        init_obj = obj(init)
        _t = time.time()

        if args.method.startswith('sa'):
            x, y = sa_optimize(obj, init, init_obj, args.temp, n_iter=args.step, step_action=args.step_size, use_tqdm=False)
            print(y)
        else:
            raise NotImplementedError

    logs = []
    for log in glob(f'{path}/tmp/{i}_*'):
        log = log.rsplit('/', 1)[-1]
        n, t, pl, cl, pu, cu = [float(j) for j in log.split('_')]
        assign, sl, su = [eval(i.strip()) for i in open(f'{path}/tmp/{log}')]
        logs.append({
            'time': t,
            'pl': pl,
            'cl': cl,
            'pu': pu,
            'cu': cu,
            'assignment': ''.join(map(str, assign)),
            'sl': [[j and ','.join(map(str, j)) for j in i] for i in sl],
            'su': str(su)
        })
    logs.sort(key=lambda x: x['time'])
    with open(f'{path}/json/{i}.json', 'w') as f:
        json.dump(logs, f, indent=2)


parallel_map(worker, [[i + args.s, p] for i, p in enumerate(problems[args.s:args.e])], num_workers=args.parallel)
