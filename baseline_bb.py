import argparse
import pickle
import time

import numpy as np
import rbfopt

from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('-n', type=int, default=25)
parser.add_argument('-i', type=int, default=0, help='i')
parser.add_argument('--method', type=str, default='bb', help='method')
parser.add_argument('--step', type=int, default=50, help='number of steps')


args = parser.parse_args()
assert args.method.startswith('bb')
settings = rbfopt.RbfoptSettings(num_cpus=16, minlp_solver_path='./bin/bonmin', nlp_solver_path='./bin/ipopt', max_evaluations=args.step)

N = args.n
problems = pickle.load(open(f'data/{N}_10k.pkl', 'rb'))
path = f'exp/{args.method}_{N}_10k'


def Nan(x):
    if np.isnan(x):
        return 1e6
    return x


solver = Solver(problems[args.i])
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
    with open(f'{path}/tmp/{args.i}_{time.time()-_t:.2f}_{pl}_{cl}_{pu}_{cu}', 'w') as f:
        f.write(f'{x.tolist()}\n{solver.state.sl}\n{solver.state.su}')
    ret = (pl + Nan(pu)) * 1e6 + Nan(cl) + Nan(cu)
    return ret


_t = time.time()
init_obj = obj(init)
_t = time.time()

n = len(init)
bb = rbfopt.RbfoptUserBlackBox(n, [0] * n, [3] * n, ['I'] * n, obj)
alg = rbfopt.RbfoptAlgorithm(settings, bb, init_node_pos=[init], init_node_val=[init_obj])
alg.optimize()
