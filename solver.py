import pickle
import time
from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from recordclass import recordclass
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from constants import LOAD, UPPER_OFFSET
from lkh import LKH
from utils import parallel_map

LOWER_SPEED = 2
UPPER_SPEED = 10

POS_X = 0
POS_Y = 1
POS_P = 2
POS_D = 3
POS_A = 4
POS_B = 5
POS_C = 6
POS_D1 = 7
POS_D2 = 8
POS_PASS = 9
POS_TOGO = 10
POS_USED = 11
POS_REMAIN = 12
POS_TW = 13


def split_zero(arr):
    assert arr[0] == 0 == arr[-1]
    ret = []
    last = 0
    for i, j in enumerate(arr[1:]):
        if j == 0:
            ret.append(arr[last:i + 2])
            last = i + 1
    return ret


def _lower_solver(args):
    relay, lkh, custom_pos, custom_demand, capacity = args
    if custom_pos.size == 0:
        return [], [], 0, 0, None
    demand = np.vstack([[0, 0, 0, 1e7, 0], custom_demand])
    p = np.vstack([relay, custom_pos])
    dist_mat = np.sqrt(np.square(p.reshape(-1, 1, 2) - p.reshape(1, -1, 2)).sum(2))
    time_mat = dist_mat / LOWER_SPEED
    sol = list(lkh.solve_acvrpspdtw(dist_mat, time_mat, demand, capacity, len(custom_pos), scale=1e5, single_courier=True, always=True))
    assert sol
    penalty, cost, sol = min(([i.penalty, i.cost, i.solution] for i in sol), key=lambda x: x[:2])
    if penalty == 0:
        rs = []
        route = []
        assert sol[0] == sol[-1] == 0
        for i in sol[1:]:
            if i:
                route.append(i)
            else:
                rs.append(route)
                route = []
        ds = []
        ds_id = []
        t = 0
        ind = np.arange(len(demand))
        for route in rs:
            p, d, a, b, c = demand[route[0]]
            # assert t <= b
            t = max(t, a - time_mat[0, route[0]])
            start = t
            last = 0
            for i in route + [0]:
                p, d, a, b, c = demand[i]
                t += time_mat[last, i]
                # assert t <= b
                if t > b:
                    pickle.dump((args, (dist_mat, time_mat, demand, capacity, len(custom_pos)), sol), open(f'err_{time.time()}.pkl', 'wb'))
                t = max(t, a) + c
                last = i
            end = t
            x = ind[route]
            y = demand[route, :2]
            p, d = y.sum(0).tolist()
            if p > 0:
                ds_id.append(x[y[:, 0] > 0])
                ds.append([p, 0, end + UPPER_OFFSET, 1e7, LOAD * p])
            if d > 0:
                ds_id.append(x[y[:, 1] > 0])
                ds.append([0, d, 0, start + UPPER_OFFSET, LOAD * d])
        return ds, ds_id, 0, cost, sol
    else:
        return [], [], penalty, cost, sol


SolverState = recordclass('SolverState', ['pl', 'cl', 'sl', 'pu', 'cu', 'su'])


class Solver():
    def __init__(self, problem, device=None, k_cost=1, k_cost_upper=1, k_penalty=1, use_incumbent_reward=False, assign_init='near', use_neg_cost=False, always_eval_upper=False, always_use_upper=False, upper_vehicle=4, lower_capacity=10, courier_cost=0, upper_trial=1000, lower_trial=100, seed=43):
        self.nx = problem['nx']
        self.ny = problem['ny']
        self.n = self.nx * self.ny
        self.relay_pos = problem['relay_pos']
        self.relay_dist = problem['relay_dist']
        cr = problem['custom_region']
        self.custom_pos = [problem['custom_pos'][cr == i] for i in range(self.n)]
        self.custom_demand = [problem['custom_demand'][cr == i] for i in range(self.n)]
        if assign_init == 'near':
            self.assignment = [np.square(
                p.reshape(-1, 1, 2) - self.relay_pos[1 + i * 4:5 + i * 4].reshape(1, -1, 2)
            ).sum(2).argmin(1) for i, p in enumerate(self.custom_pos)]
        elif assign_init == 'random':
            self.assignment = [np.random.randint(low=0, high=3, size=len(i)) for i in self.custom_pos]
        elif assign_init in ['kmeans', 'gmm']:
            self.assignment = []
            for i, cp in enumerate(self.custom_pos):
                rp = self.relay_pos[1 + 4 * i:5 + 4 * i]
                if assign_init == 'kmeans':
                    cl = KMeans(len(rp), random_state=seed).fit_predict(np.vstack([rp, cp]))[len(rp):]
                else:  # assign_init=='gmm'
                    cl = GaussianMixture(len(rp), random_state=seed).fit_predict(np.vstack([rp, cp]))[len(rp):]
                assigned = [False] * len(rp)
                assign = np.zeros(len(cp)) - 1
                for i, _ in sorted(Counter(cl).items(), key=lambda x: -x[1]):
                    mask = cl == i
                    d = np.sqrt(np.sum(np.square(cp[mask].reshape(-1, 1, 2) - rp.reshape(1, -1, 2)), 2))
                    for j in np.argsort(np.sum(d, 0)):
                        if not assigned[j]:
                            assign[mask] = j
                            assigned[j] = True
                            break
                    else:
                        assert False
                assert np.all(assign >= 0)
                self.assignment.append(assign)
        else:
            raise NotImplementedError
        self.state = SolverState(
            np.zeros(self.n) + np.nan,
            np.zeros(self.n) + np.nan,
            [None] * self.n,
            np.nan,
            np.nan,
            None)
        self.demand_upper = [None] * self.n
        self.demand_ids = [None] * self.n
        self.demand_detail_ids = [None] * self.n
        self.upper_vehicle = upper_vehicle
        self.lower_capacity = lower_capacity
        self.courier_cost = courier_cost
        self.upper_trial = upper_trial
        self.lower_trial = lower_trial
        self.k_cost = k_cost
        self.k_cost_upper = k_cost_upper
        self.k_penalty = k_penalty
        self.use_incumbent_reward = use_incumbent_reward
        self.use_neg_cost = use_neg_cost
        self.always_use_upper = always_use_upper
        if always_eval_upper and not always_use_upper:
            print('WARN: always eval upper but not used')
        self.always_eval_upper = always_eval_upper and always_use_upper
        if device is not None:
            self.t_dist = torch.tensor(self.relay_dist, dtype=torch.float32, device=device)
            self.t_upper = torch.tensor(self.relay_pos, dtype=torch.float32, device=device)
            self.t_lower = [
                torch.tensor(
                    [[x, y, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for x, y in self.relay_pos[1 + i * 4:5 + i * 4]] +
                    [[x, y, p, d, a, b, c, 0, 0, 0, 0, 0, 0, 0] for (x, y), (p, d, a, b, c) in zip(self.custom_pos[i], self.custom_demand[i])], dtype=torch.float32, device=device)
                for i in range(self.n)]
            for i in range(4):
                self.eval_lower(i)
            self.eval_upper()
            self.prepare_torch()
            self.init_state = deepcopy(self.state)
            self.best_state = deepcopy(self.state)
            self.best_assign = deepcopy(self.assignment)
            self.record_cost()

    def record_cost(self):
        if (
                sum(self.state.pl) < sum(self.best_state.pl) or
                sum(self.state.pl) == 0 and (
                    self.state.pu < self.best_state.pu or
                    self.state.pu == 0 and sum(self.state.cl) + self.state.cu < sum(self.best_state.cl) + self.best_state.cu
                )
        ):
            self.best_state = deepcopy(self.state)
            self.best_assign = deepcopy(self.assignment)

    def reset_best(self):
        self.state = deepcopy(self.best_state)
        self.assignment = deepcopy(self.best_assign)
        self.prepare_torch()

    def prepare_torch(self, index=None):
        if index is None:
            indices = range(self.n)
        else:
            indices = [index]
        for index in indices:
            tl, s = self.t_lower[index], self.state.sl[index]
            cp = self.custom_pos[index]
            cd = self.custom_demand[index]
            rp = self.relay_pos[1 + index * 4:5 + index * 4]
            for ir, s__ in enumerate(s):
                if s__ is None:
                    continue
                for s_ in split_zero(s__):
                    id_ = [i - 1 for i in s_[1:-1]]
                    d = [np.linalg.norm(rp[ir] - cp[id_[0]])] + np.linalg.norm(cp[id_[:-1]] - cp[id_[1:]], axis=1).tolist() + [np.linalg.norm(rp[ir] - cp[id_[-1]])]
                    d1 = np.cumsum(d)
                    d2 = np.cumsum(d[::-1])[::-1]
                    cd_ = cd[id_]
                    cap = np.cumsum(cd_[:, 0])
                    cap[:-1] += np.cumsum(cd_[:0:-1, 1])[::-1]
                    t = 0
                    for _id, _d1, _d2, _pass, _togo, _used, _remain in zip(id_, d, d[1:], d1, d2[1:], cap, self.lower_capacity - cap):
                        t += _d1 / LOWER_SPEED
                        ta, tb, tc = cd[_id][2:]
                        tl_ = tl[_id]
                        tl_[POS_TW] = max(0, ta - t) + min(0, tb - t)
                        t = max(t, ta) + tc
                        tl_[POS_D1] = _d1
                        tl_[POS_D2] = _d2
                        tl_[POS_PASS] = _pass
                        tl_[POS_TOGO] = _togo
                        tl_[POS_USED] = _used
                        tl_[POS_REMAIN] = _remain

    def sample_action(self, index, step_explore):
        raise NotImplementedError

    def step(self, index, assignment):
        old_state = deepcopy(self.state)
        if len(assignment) == 2:
            self.assignment[index][assignment[0]] = assignment[1]
            self.eval_lower(index)
        assert np.all(~np.isnan(self.state.pl))
        assert np.all(~np.isnan(old_state.pl))
        if self.use_incumbent_reward:
            raise NotImplementedError
        elif self.use_neg_cost:
            pl = self.state.pl[index]
            cl = self.state.cl[index]
            if pl == 0:
                assert not np.isnan(cl)
                reward = 1 - min(1, cl * self.k_cost)
            else:
                reward = -min(1, pl * self.k_penalty)
            # call solver in the last region
            if index == self.n - 1 or self.always_eval_upper:
                self.eval_upper()
                self.record_cost()
            # consider upper reward
            if index == self.n - 1 or self.always_use_upper:
                pu = self.state.pu
                cu = self.state.cu
                if np.isnan(pu):
                    reward += -1
                elif pu == 0:
                    assert not np.isnan(cu)
                    reward += 1 - min(1, cu * self.k_cost)
                else:
                    reward += -min(1, pu * self.k_penalty)
                reward *= 0.5
        else:
            if self.state.pl[index] == 0:
                if old_state.pl[index] == 0:
                    reward = max(-1, min(1, (old_state.cl[index] - self.state.cl[index]) * self.k_cost))
                    assert not np.isnan(reward)
                else:
                    reward = 0
            else:
                reward = -min(1, self.state.pl[index] * self.k_penalty)
                assert not np.isnan(reward)
            if index == self.n - 1 or self.always_eval_upper:
                self.eval_upper()
                self.record_cost()
            if index == self.n - 1 or self.always_use_upper:
                if np.isnan(self.state.pu):
                    reward += -1
                elif self.state.pu == 0:
                    if old_state.pu == 0:
                        assert not np.isnan(old_state.cu)
                        reward += max(-1, min(1, (old_state.cu - self.state.cu) * self.k_cost)) * self.k_cost_upper
                        assert not np.isnan(reward)
                else:
                    reward += -min(1, self.state.pu * self.k_penalty)
                    assert not np.isnan(reward)
        self.prepare_torch(index)
        return reward

    def eval_lower(self, i):
        lkh = LKH(trial=self.lower_trial)
        custom_pos = self.custom_pos[i]
        custom_demand = self.custom_demand[i]
        assignment = self.assignment[i]
        demands = []
        demand_ids = []
        demand_detail_ids = []
        cost = 0
        penalty = 0
        pr = []
        sols = []
        true_id = []
        ind = np.arange(len(assignment))
        for ir, r in enumerate(self.relay_pos[1 + i * 4:5 + i * 4]):
            s = assignment == ir
            true_id.append(ind[s])
            pr.append((r, lkh, custom_pos[s], custom_demand[s], self.lower_capacity))
        for ir, (ds, ds_id, p, c, s) in enumerate(parallel_map(_lower_solver, pr, use_tqdm=False)):
            penalty += p
            cost += c
            ti = true_id[ir]
            if ds:
                demands += ds
                demand_ids += [ir] * len(ds)
                demand_detail_ids += (ti[i - 1].tolist() for i in ds_id)
                cost += self.courier_cost
            sols.append(s if s is None else [i and ti[i - 1] + 1 for i in s])
        self.state.pl[i] = penalty
        self.state.cl[i] = cost if penalty == 0 else np.nan
        self.state.sl[i] = sols
        self.demand_upper[i] = demands
        self.demand_ids[i] = demand_ids
        self.demand_detail_ids[i] = demand_detail_ids

    def eval_upper(self):
        if np.any(self.state.pl):
            self.state.cu = np.nan
            self.state.pu = np.nan
            return False
        demands = sum(self.demand_upper, [[0, 0, 0, 1e7, 0]])
        demand_ids = [0] + [1 + i * 4 + k for i, j in enumerate(self.demand_ids) for k in j]
        id_map = {i: j for i, j in enumerate(demand_ids)}
        dist_mat = np.zeros((len(id_map),) * 2)
        for i, ii in id_map.items():
            for j, jj in id_map.items():
                dist_mat[i, j] = self.relay_dist[ii, jj]
        sol = list(LKH(trial=self.upper_trial).solve_acvrpspdtw(dist_mat, dist_mat / UPPER_SPEED, demands, 1e7, self.upper_vehicle, scale=1e5, always=True))
        assert sol
        self.state.pu, self.state.cu, sol = min(([i.penalty, i.cost, i.solution] for i in sol), key=lambda x: x[:2])
        if self.state.pu == 0:
            assert sol is not None
            self.state.su = [sol, demand_ids, self.demand_detail_ids]
        else:
            self.state.cu = np.nan
        return True

    def show_image(self, exaggerate=False):
        colors = ['r', 'g', 'b', 'gray']
        if exaggerate:
            offset = np.array([0, 0, 0, 100, 100, 0, 100, 100]).reshape(-1, 2)
            for c, a, o in zip(self.custom_pos, self.assignment, offset):
                cc = c + o
                for i in range(4):
                    plt.scatter(*cc[a == i].T, color=colors[i])
            border = np.array([[5, 5], [295, 5], [295, 295], [5, 295], [5, 5]])
            relay = np.array([[200, 295], [295, 100], [100, 5], [5, 200]])
            for x in [0, 400]:
                for y in [0, 400]:
                    plt.plot(*(border + np.array([x, y])).T, color='k', linewidth=1)
                    for r, c in zip(relay, colors):
                        plt.scatter(*(r + np.array([x, y])), marker='x', color=c)
            plt.scatter(350, 350, 100, color='k', marker='*', zorder=10)
        else:
            for c, a in zip(self.custom_pos, self.assignment):
                for i in range(4):
                    plt.scatter(*c[a == i].T, 1, color=colors[i])
            for i, (c, ss) in enumerate(zip(self.custom_pos, self.state.sl or [])):
                r = self.relay_pos[1 + 4 * i: 5 + 4 * i]
                for i, s in enumerate(ss):
                    if s is not None:
                        plt.plot(
                            *zip(*[c[j - 1] if j else r[i] for j in s]),
                            color=colors[i],
                            alpha=0.5,
                        )
            border = np.array([[5, 5], [295, 5], [295, 295], [5, 295], [5, 5]])
            relay = np.array([[200, 295], [295, 100], [100, 5], [5, 200]])
            for x in [0, 300]:
                for y in [0, 300]:
                    plt.plot(*(border + np.array([x, y])).T, color='k', linewidth=1)
                    for r, c in zip(relay, colors):
                        plt.scatter(*(r + np.array([x, y])), marker='x', color=c)
            plt.scatter(300, 300, 100, color='k', marker='*', zorder=10)
        plt.axis('equal')
        plt.gcf().set_size_inches(4, 4)

    def save_image(self, path):
        self.show_image()
        plt.savefig(path)
        plt.clf()

    def eval_all(self):
        parallel_map(lambda i: self.eval_lower(i), range(4), num_workers=4, use_tqdm=False)
        self.eval_upper()
        return sum(self.state.pl) + self.state.pu, sum(self.state.cl) + self.state.cu
