import re
import subprocess
from collections import namedtuple

import numpy as np

Log = namedtuple('Log', ['trial', 'penalty', 'cost', 'time', 'solution'])


class LKH():
    def __init__(self, *, trial=1000, seed=43, lkh_path='./bin/LKH'):
        self.trial = trial
        self.seed = seed
        self.lkh_path = lkh_path

    def solve_acvrpspdtw(self, dist_mat, time_mat, demand, capacity, n_vehicle, *, l_opt_limit=6, scale=1, trial=-1, seed=-1, log_only=False, always=False, single_courier=False):
        if trial <= 0:
            trial = self.trial
        if seed < 0:
            seed = self.seed
        dist_mat = (np.array(dist_mat) * scale).astype(int)
        time_mat = (np.array(time_mat) * scale).astype(int)
        demand = (np.array(demand) * scale).astype(int)
        assert demand.shape[0] == dist_mat.shape[0] == time_mat.shape[0]
        n_vehicle = max(1, min(demand.shape[0] - 1, n_vehicle))
        capacity = int(capacity * scale)
        always = always and not log_only
        if type(l_opt_limit) is not int or not 2 <= l_opt_limit <= 6:
            raise ValueError()
        f_par = f'''SPECIAL
PROBLEM_FILE = -
MAX_TRIALS = {trial}
{'ALWAYS_WRITE_OUTPUT' if always else ''}
RUNS = 1
POPULATION_SIZE = 10
TRACE_LEVEL = 1
{'WRITE_SOLUTION_TO_LOG' if not log_only else ''}
{'SINGLE_COURIER' if single_courier else ''}
L_OPT_LIMIT = {l_opt_limit}
SEED = {seed}
$$$
'''
        f_vrp = f'''NAME : xxx
TYPE : VRPSPDTW
DIMENSION : {dist_mat.shape[0]}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
CAPACITY : {capacity}
VEHICLES : {n_vehicle}
EDGE_WEIGHT_SECTION
'''
        f_vrp += '\n'.join(' '.join(str(j) for j in i) for i in dist_mat) + '\n'
        f_vrp += 'EDGE_TIME_SECTION\n'
        f_vrp += '\n'.join(' '.join(str(j) for j in i) for i in time_mat) + '\n'
        f_vrp += 'PICKUP_AND_DELIVERY_SECTION\n'
        for i, (p, d, a, b, c) in enumerate(demand):
            f_vrp += f'{i+1} 0 {a} {b} {c} {p} {d}\n'
        f_vrp += 'DEPOT_SECTION\n1\nEOF\n$$$\n'
        last = None
        proc = subprocess.Popen(f'stdbuf -oL {self.lkh_path} -', shell=True, encoding='utf8', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write(f_par)
        proc.stdin.write(f_vrp)
        proc.stdin.flush()
        try:
            for l in proc.stdout:
                if log_only:
                    ret = re.findall(r'^\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ', l)
                    if ret:
                        i, p, c, t = ret[0]
                        yield Log(int(i), float(p) / scale, float(c) / scale, float(t), None)
                else:
                    if l.startswith('Solution:'):
                        if last[0] == '*':
                            i, p, c, t = re.findall(r'\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ', last)[0]
                            sol = [int(k) for k in l.split(' ', 1)[1].split(',')]
                            yield Log(int(i), float(p) / scale, float(c) / scale, float(t), sol)
                    last = l
        finally:
            proc.kill()
            proc.wait()

    def solve_acvrp(self, dist_mat, demand, capacity, n_vehicle, *, l_opt_limit=6, scale=1, trial=-1, seed=-1, log_only=False, always=False):
        if trial <= 0:
            trial = self.trial
        if seed < 0:
            seed = self.seed
        dist_mat = (np.array(dist_mat) * scale).astype(int)
        demand = (np.array(demand) * scale).astype(int).reshape(-1)
        assert demand.shape[0] + 1 == dist_mat.shape[0]
        n_vehicle = max(1, min(demand.shape[0], n_vehicle))
        capacity = int(capacity * scale)
        always = always and not log_only
        if type(l_opt_limit) is not int or not 2 <= l_opt_limit <= 6:
            raise ValueError()
        f_par = f'''SPECIAL
PROBLEM_FILE = -
MAX_TRIALS = {trial}
{'ALWAYS_WRITE_OUTPUT' if always else ''}
RUNS = 1
POPULATION_SIZE = 10
TRACE_LEVEL = 1
{'WRITE_SOLUTION_TO_LOG' if not log_only else ''}
L_OPT_LIMIT = {l_opt_limit}
SEED = {seed}
$$$
'''
        f_vrp = f'''NAME : xxx
TYPE : ACVRP
DIMENSION : {dist_mat.shape[0]}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
CAPACITY : {capacity}
VEHICLES : {n_vehicle}
EDGE_WEIGHT_SECTION
'''
        f_vrp += '\n'.join(' '.join(str(j) for j in i) for i in dist_mat) + '\n'
        f_vrp += 'DEMAND_SECTION\n'
        for i, d in enumerate([0] + demand.tolist()):
            f_vrp += f'{i+1} {d}\n'
        f_vrp += 'DEPOT_SECTION\n1\nEOF\n$$$\n'
        last = None
        proc = subprocess.Popen(f'stdbuf -oL {self.lkh_path} -', shell=True, encoding='utf8', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write(f_par)
        proc.stdin.write(f_vrp)
        proc.stdin.flush()
        try:
            for l in proc.stdout:
                if log_only:
                    ret = re.findall(r'^\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ', l)
                    if ret:
                        i, p, c, t = ret[0]
                        yield Log(int(i), float(p) / scale, float(c) / scale, float(t), None)
                else:
                    if l.startswith('Solution:'):
                        if last[0] == '*':
                            i, p, c, t = re.findall(r'\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ', last)[0]
                            sol = [int(k) for k in l.split(' ', 1)[1].split(',')]
                            yield Log(int(i), float(p) / scale, float(c) / scale, float(t), sol)
                    last = l
        finally:
            proc.kill()
            proc.wait()

    def solve_pdptw(self, dist_mat, demand, capacity, n_vehicle, *, l_opt_limit=6, scale=1, trial=-1, seed=-1, log_only=False, always=False):
        if trial <= 0:
            trial = self.trial
        if seed < 0:
            seed = self.seed
        dist_mat = (np.array(dist_mat) * scale).astype(int)
        demand = np.array(demand)
        demand[:, [0, 3, 4, 5]] *= scale
        demand = demand.astype(int)
        assert demand.shape[0] == dist_mat.shape[0]
        n_vehicle = max(1, min(demand.shape[0] - 1, n_vehicle))
        capacity = int(capacity * scale)
        always = always and not log_only
        if type(l_opt_limit) is not int or not 2 <= l_opt_limit <= 6:
            raise ValueError()
        f_par = f'''SPECIAL
PROBLEM_FILE = -
MAX_TRIALS = {trial}
{'ALWAYS_WRITE_OUTPUT' if always else ''}
RUNS = 1
POPULATION_SIZE = 10
TRACE_LEVEL = 1
{'WRITE_SOLUTION_TO_LOG' if not log_only else ''}
L_OPT_LIMIT = {l_opt_limit}
SEED = {seed}
$$$
'''
        f_vrp = f'''NAME : xxx
TYPE : PDPTW
DIMENSION : {dist_mat.shape[0]}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
CAPACITY : {capacity}
VEHICLES : {n_vehicle}
EDGE_WEIGHT_SECTION
'''
        f_vrp += '\n'.join(' '.join(str(j) for j in i) for i in dist_mat) + '\n'
        f_vrp += 'PICKUP_AND_DELIVERY_SECTION\n'
        for i, (d, f, t, a, b, c) in enumerate(demand):
            if i == 0:
                assert d == f == t == a == c == 0
            assert (f == 0) if d > 0 else (t == 0)
            f_vrp += f'{i+1} {d} {a} {b} {c} {0 if f==0 else f+1} {0 if t==0 else t+1}\n'
        f_vrp += 'DEPOT_SECTION\n1\nEOF\n$$$\n'
        last = None
        proc = subprocess.Popen(f'stdbuf -oL {self.lkh_path} -', shell=True, encoding='utf8', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write(f_par)
        proc.stdin.write(f_vrp)
        proc.stdin.flush()
        try:
            for l in proc.stdout:
                if log_only:
                    ret = re.findall(r'^\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ', l)
                    if ret:
                        i, p, c, t = ret[0]
                        yield Log(int(i), float(p) / scale, float(c) / scale, float(t), None)
                else:
                    if l.startswith('Solution:'):
                        if last[0] == '*':
                            i, p, c, t = re.findall(r'\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ', last)[0]
                            sol = [int(k) for k in l.split(' ', 1)[1].split(',')]
                            yield Log(int(i), float(p) / scale, float(c) / scale, float(t), sol)
                    last = l
        finally:
            proc.kill()
            proc.wait()
