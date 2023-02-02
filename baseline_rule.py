import argparse
import pickle

import numpy as np

from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('-n', type=int, default=25)
parser.add_argument('-i', type=int, default=0, help='i')
parser.add_argument('--method', type=str, default='near', help='method')
parser.add_argument('--init', choices=['random', 'near'], default='near')


args = parser.parse_args()

N = args.n
problems = pickle.load(open(f'data/{N}_10k.pkl', 'rb'))
path = f'exp/{args.method}_{N}_10k'


def Nan(x):
    if np.isnan(x):
        return 1e6
    return x


solver = Solver(problems[args.i], assign_init=args.method)
init = np.hstack(solver.assignment)
ind = np.cumsum([0] + [len(i) for i in solver.assignment])
ind = [slice(i, j) for i, j in zip(ind, ind[1:])]


def obj(x):
    x = x.astype(int)
    solver.assignment = [x[i] for i in ind]
    solver.eval_all()
    pl, cl, pu, cu = sum(solver.state.pl), sum(solver.state.cl), solver.state.pu, solver.state.cu
    with open(f'{path}/tmp/{args.i}_0.0_{pl}_{cl}_{pu}_{cu}', 'w') as f:
        f.write(f'{x.tolist()}\n{solver.state.sl}\n{solver.state.su}')
    ret = (pl + Nan(pu)) * 1e6 + Nan(cl) + Nan(cu)
    return ret


init_obj = obj(init)
