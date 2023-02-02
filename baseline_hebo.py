import argparse
import os
import pickle
import time

import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('-n', type=int, default=25)
parser.add_argument('-i', type=int, default=0, help='i')
parser.add_argument('--method', type=str, default='bo', help='method')
parser.add_argument('--step', type=int, default=50, help='number of steps')

args = parser.parse_args()
assert args.method.startswith('bo')

N = args.n
problems = pickle.load(open(f'data/{N}_10k.pkl', 'rb'))
path = f'exp/{args.method}_{N}_10k'


def Nan(x):
    if np.isnan(x):
        return 1e6
    return x


def bo_optimize(func, x, y, range_x=4, step=30):
    n = len(x)
    min_y = y
    space = DesignSpace().parse([
        {'name': f'x{i}', 'type': 'int', 'lb': 0, 'ub': range_x - 1} for i in range(n)
    ])
    hebo_seq = HEBO(space, model_name='gpy', rand_sample=4, model_config={'num_epoch': 10})
    hebo_x = hebo_seq.suggest()
    hebo_x[:] = x
    for _ in range(step):
        hebo_seq.observe(hebo_x, y.reshape(-1, 1))
        hebo_x = hebo_seq.suggest(n_suggestions=1)
        y = func(hebo_x.to_numpy().reshape(-1))
        min_y = min(min_y, y)
    return min_y


def main():
    i = args.i
    p = problems[i]
    if os.path.exists(f'{path}/json/{i}.json'):
        return
    solver = Solver(p, assign_init=args.init)
    init = np.hstack(solver.assignment)
    ind = np.cumsum([0] + [len(i) for i in solver.assignment])
    ind = [slice(i, j) for i, j in zip(ind, ind[1:])]

    def obj(x):
        x = x.astype(int)
        if args.add:
            x = (x + init) % 4
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

    y = bo_optimize(obj, init, np.array([init_obj]), step=args.step)


if __name__ == '__main__':
    main()
