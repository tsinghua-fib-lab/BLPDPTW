import argparse
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
from bisect import bisect_right
from distutils.util import strtobool
from glob import glob
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from solver import Solver
from utils import parallel_map, wheel


def uniq(arr):
    out = arr[:1]
    for i, j in zip(arr, arr[1:]):
        if i != j:
            out.append(j)
    return out


class ParallelSolvers():
    def __init__(self, solvers, num_envs, num_parallel, episode, curriculum):
        self.solvers = solvers
        assert num_envs <= len(solvers)
        self.solver_step_cnt = torch.zeros(len(solvers), dtype=torch.long)
        self.i = 0
        self.n = solvers[0].n
        self.episode = episode
        self.curriculum = curriculum
        self.wheel = wheel(len(solvers), num_envs)
        self.solver_index = next(self.wheel)
        self.current_solvers = [solvers[i] for i in self.solver_index]
        self.device = solvers[0].t_dist.device
        self._d = torch.zeros(num_envs, device=self.device)
        self.num_parallel = num_parallel

    def observe(self):
        cs = self.current_solvers
        return [
            torch.stack([s.t_dist for s in cs]),
            torch.stack([s.t_upper for s in cs]),
            [s.state.su for s in cs],
            [s.t_lower[self.i] for s in cs],
            [s.state.sl for s in cs],
            torch.tensor([self.i] * len(cs), dtype=torch.long, device=self.device)
        ]

    def current_assignment(self):
        return [torch.from_numpy(i.assignment[self.i]).to(self.device) for i in self.current_solvers]

    def step(self, action):
        cs = self.current_solvers
        reward = torch.tensor(
            parallel_map(lambda x: x[0].step(self.i, x[1].T.detach().cpu().numpy()), zip(cs, action), self.num_parallel, use_tqdm=False),
            dtype=torch.float32, device=self.device)
        self.solver_step_cnt[self.solver_index] += 1
        if self.episode > 0:
            d_ = self.solver_step_cnt[self.solver_index] % self.episode == 0
            done = d_.to(self.device, torch.float)
        else:
            if self.curriculum > 0:
                d_ = self.solver_step_cnt[self.solver_index] % self.curriculum == 0
            done = self._d
        self.i = (self.i + 1) % self.n
        obs = self.observe()
        if self.episode > 0 or self.curriculum > 0:
            for s, d in zip(cs, d_):
                if d:
                    s.reset_best()
        if self.i == 0:
            self.solver_index = next(self.wheel)
            self.current_solvers = [self.solvers[i] for i in self.solver_index]
            return reward, obs, done, self.observe()
        return reward, obs, done, None

    def sample_actions(self, step_explore):
        return [s.sample_action(self.i, step_explore) for s in self.current_solvers]


class ObsIndexer():
    def __init__(self, obs):
        self.l = np.cumsum([0] + [len(i[0]) for i in obs])
        self.obs = obs

    def __getitem__(self, index):
        out = []
        for i in index:
            ii = self.l[bisect_right(self.l, i) - 1]
            i -= ii
            out.append([j[i] for j in self.obs[ii]])
        a, b, c, d, e, f = zip(*out)
        return [
            torch.stack(a),
            torch.stack(b),
            c, d, e,
            torch.stack(f)
        ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=43, help="seed of the experiment")
    parser.add_argument("--cuda", type=int, default=-1)

    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--share-base", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Share base network for actor and critic")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gae-lambda", type=float, default=0.9,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
                        help="the K epochs to update the policy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--clip-coef", type=float, default=0.25,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")

    parser.add_argument("--load", type=str, default="", help='model checkpoint')
    parser.add_argument("--data", type=str, default="data/400_10k.pkl")
    parser.add_argument('--cost-norm', type=float, default=40000, help='cost is divided by this value')
    parser.add_argument('--cost-upper-coeff', type=float, default=1, help='coeff of upper cost')
    parser.add_argument('--penalty-norm', type=float, default=100000, help='penalty is divided by this value')
    parser.add_argument('--courier_cost', type=float, default=0, help='cost sending each courier')
    parser.add_argument('--courier_capacity', type=float, default=10)
    parser.add_argument('--incumbent-reward', action='store_true', help='to use difference between incumbent as reward')
    parser.add_argument('--use-neg-cost', action='store_true', help='reward = -cost')
    parser.add_argument('--always-use-upper', action='store_true', help='always use upper cost and penalty in reward')
    parser.add_argument('--always-eval-upper', action='store_true', help='always evaluate upper cost and penalty')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--num-instances', type=int, default=1000, help='# of instances to use')
    parser.add_argument('--num-envs', type=int, default=64, help='# of environments')
    parser.add_argument('--num-parallel', type=int, default=16, help='# of parallel environments')
    parser.add_argument('--save-interval', type=int, default=10, help='checkpoint save interval')

    parser.add_argument('--embed-dim', type=int, default=64, help='embedding dimension of orders')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dist-hidden-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--curriculum', type=int, default=0, help='reset interval for curriculum learning')
    parser.add_argument('--episode', type=int, default=0, help='enable episodic learning')
    parser.add_argument('--num-steps', type=int, default=1, help='env rollout steps')
    parser.add_argument('--step-explore', type=int, default=16, help='maximum changed assignments in exploration')
    parser.add_argument('--step-action', type=int, default=8, help='maximum changed assignments in action')
    parser.add_argument('--no-lstm', action='store_true', help='do not use lstm module')
    parser.add_argument('--no-mha', action='store_true', help='do not use mha module')
    args = parser.parse_args()
    args.num_envs = min(args.num_envs, args.num_instances)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


class MHA_Layer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_1 = nn.LayerNorm(embed_dim)
        if mlp_hidden_dim is None:
            mlp_dim = (embed_dim, embed_dim * 4, embed_dim)
        else:
            mlp_dim = (embed_dim,) + mlp_hidden_dim + (embed_dim,)
        self.mlp = nn.Sequential(
            *sum(([
                nn.Linear(i, j),
                nn.ReLU(),
                nn.Dropout(dropout)
            ] for i, j in zip(mlp_dim, mlp_dim[1:-1])), []),
            nn.Linear(mlp_dim[-2], mlp_dim[-1]),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.ln_1(x + self.mha(x, x, x)[0])
        x = self.ln_2(x + self.mlp(x))
        return x


class MHAD_Layer_(nn.Module):
    def __init__(self, embed_dim, num_heads, dist_hidden_dim, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(4))
        self.mix = nn.ModuleList(nn.Sequential(
            nn.Linear(2, dist_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dist_hidden_dim, dist_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dist_hidden_dim, 1)
        ) for _ in range(num_heads))

    def forward(self, query, key, value, dist_mat):
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.stack([m(torch.stack([a, dist_mat], dim=-1)).squeeze(-1) for m, a in zip(self.mix, attn.transpose(0, 1))]).transpose(0, 1)
        x = torch.matmul(attn, value).transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class MHAD_Layer(nn.Module):
    def __init__(self, embed_dim, num_heads, dist_hidden_dim, mlp_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.mha = MHAD_Layer_(embed_dim, num_heads, dist_hidden_dim, dropout=dropout)
        self.ln_1 = nn.LayerNorm(embed_dim)
        if mlp_hidden_dim is None:
            mlp_dim = (embed_dim, embed_dim * 4, embed_dim)
        else:
            mlp_dim = (embed_dim,) + mlp_hidden_dim + (embed_dim,)
        self.mlp = nn.Sequential(
            *sum(([
                nn.Linear(i, j),
                nn.ReLU(),
                nn.Dropout(dropout)
            ] for i, j in zip(mlp_dim, mlp_dim[1:-1])), []),
            nn.Linear(mlp_dim[-2], mlp_dim[-1]),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, d):
        x = self.ln_1(x + self.mha(x, x, x, d))
        x = self.ln_2(x + self.mlp(x))
        return x


class MHAD(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dist_hidden_dim, dropout=0.1):
        super().__init__()
        self.mhad = nn.ModuleList(MHAD_Layer(embed_dim, num_heads, dist_hidden_dim, dropout=dropout) for _ in range(num_layers))

    def forward(self, x, d):
        for l in self.mhad:
            x = l(x, d)
        return x


def make_mlp(*shape, dropout, sigma=0):
    ls = [nn.Linear(i, j) for i, j in zip(shape, shape[1:])]
    if sigma > 0:
        for l in ls:
            nn.init.orthogonal_(l.weight, 2**0.5)
            nn.init.constant_(l.bias, 0)
        nn.init.orthogonal_(ls[-1].weight, sigma)
    return nn.Sequential(
        *sum(([
            l,
            nn.ReLU(),
            nn.Dropout(dropout),
        ] for l in ls[:-1]), []),
        ls[-1]
    )


class Base(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dist_hidden_dim, dropout=0.1, use_embed=True, use_state=True, use_mha=True, use_lstm=True):
        super().__init__()
        self.relay_norm = nn.BatchNorm1d(2)
        self.relay_embed = nn.Linear(2, embed_dim)
        if use_mha:
            self.relay_mha = MHAD(embed_dim, num_heads, num_layers, dist_hidden_dim, dropout=dropout)
        if use_lstm:
            self.relay_lstm = nn.LSTM(embed_dim, embed_dim // 2, num_layers, batch_first=True, bidirectional=True)
        self.region_norm = nn.BatchNorm1d(14)
        self.region_embed = nn.Linear(14, embed_dim)
        if use_mha:
            self.region_mha = nn.Sequential(
                *(MHA_Layer(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers))
            )
        if use_lstm:
            self.region_lstm = nn.LSTM(embed_dim, embed_dim // 2, num_layers, batch_first=True, bidirectional=True)
        self.relay_mlp = make_mlp(2 * embed_dim, 4 * embed_dim, embed_dim, dropout=dropout)
        self.output_mlp = make_mlp(4, 16, 4, dropout=dropout, sigma=0.01)
        self.use_embed = use_embed
        self.use_state = use_state
        assert use_embed or use_state
        self.use_lstm = use_lstm
        self.use_mha = use_mha
        print(f'Use LSTM: {use_lstm}')
        print(f'Use MHA: {use_mha}')

    def forward(self, obs):
        dist, upper, upper_sol, lowers, lower_sols, steps = obs
        upper = self.relay_embed(self.relay_norm(upper.view(-1, upper.size(-1))).view(upper.shape))
        if self.use_mha:
            relay_1 = self.relay_mha(upper, dist)
        else:
            relay_1 = upper
        relay_3 = []
        if self.use_state:
            states = []
        else:
            states = [[]] * len(upper)
        for r, s in zip(relay_1, upper_sol):
            if s is None:
                relay_3.append(r.new_zeros((r.size(0) - 1, r.size(1))))
                if self.use_state:
                    states.append([torch.zeros_like(r[0])])
            else:
                if self.use_lstm:
                    s = uniq(np.array(s[1])[s[0]].tolist())
                    inv = {j: i for i, j in enumerate(s)}
                    inv = [inv.get(i, -1) for i in range(17)]
                    out = self.relay_lstm(r[s])[0]
                    relay_3.append(torch.stack([out[i] if i >= 0 else torch.zeros_like(out[0]) for i in inv]))
                    if self.use_state:
                        states.append([out[-1]])
                else:
                    relay_3.append(r)
                    if self.use_state:
                        states.append([r.mean(0)])
        lowers_out = []
        for step, relay_3_, lower, sols, state in zip(steps, relay_3, lowers, lower_sols, states):
            lower = self.region_embed(self.region_norm(lower.view(-1, lower.size(-1))).view(lower.shape))
            if self.use_mha:
                lower = self.region_mha(lower)
            relay_2 = lower[:4]
            custom = lower[4:]
            relay_4 = self.relay_mlp(torch.hstack([relay_2, relay_3_[step * 4: 4 + step * 4]]))
            if self.use_embed:
                lower_out = [[None] * 4, [None] * (len(lower) - 4)]
            for ir, (r, sol) in enumerate(zip(relay_4, sols[step])):
                if sol:
                    if self.use_lstm:
                        lower = self.region_lstm(torch.stack([r if i == 0 else custom[i - 1] for i in sol]))[0]
                        if self.use_embed:
                            lower_out[0][ir] = lower[-1]
                            for i, j in enumerate(sol):
                                if j:
                                    lower_out[1][j - 1] = lower[i]
                        if self.use_state:
                            state.append(lower[-1])
                    else:
                        s = sum(custom[i - 1] for i in sol) / len(sol)
                        if self.use_embed:
                            lower_out[0][ir] = s
                            lower_out[1] = list(custom)
                        if self.use_state:
                            state.append(s)
                else:
                    if self.use_embed:
                        lower_out[0][ir] = torch.zeros_like(custom[0])
                    if self.use_state:
                        state.append(torch.zeros_like(custom[0]))
            if self.use_embed:
                lowers_out.append([torch.stack(i) for i in lower_out])
        if self.use_state:
            states = torch.stack([torch.cat(i) for i in states])
        if self.use_embed and self.use_state:
            return lowers_out, states
        elif self.use_state:
            return states
        else:
            return lowers_out


class Agent(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dist_hidden_dim, dropout=0.1, share_base=True, use_mha=True, use_lstm=True):
        super().__init__()
        if share_base:
            self.base = Base(embed_dim, num_heads, num_layers, dist_hidden_dim, dropout, use_mha=use_mha, use_lstm=use_lstm)
        else:
            self.base_actor = Base(embed_dim, num_heads, num_layers, dist_hidden_dim, dropout, use_state=False, use_mha=use_mha, use_lstm=use_lstm)
            self.base_critic = Base(embed_dim, num_heads, num_layers, dist_hidden_dim, dropout, use_embed=False, use_mha=use_mha, use_lstm=use_lstm)
        self.actor = make_mlp(2 * embed_dim, 4 * embed_dim, 1, dropout=dropout, sigma=0.01)
        self.critic = make_mlp(5 * embed_dim, 8 * embed_dim, 1, dropout=dropout, sigma=1)
        self.share_base = share_base

    def get_value(self, obs):
        if self.share_base:
            state = self.base(obs)[1]
        else:
            state = self.base_critic(obs)
        return self.critic(state)

    def get_action_and_value(self, obs, action=None, topk=2, sample=True, current_assign=None, least_action=2):
        if self.share_base:
            embed, state = self.base(obs)
        else:
            embed = self.base_actor(obs)
            state = self.base_critic(obs)
        acts = []
        log_probs = []
        for i, (relay, custom) in enumerate(embed):
            score = self.actor(torch.hstack([custom.repeat(1, len(relay)).view(-1, custom.size(1)), relay.repeat(len(custom), 1)])).view(len(custom), len(relay))
            s_ = score.softmax(1)
            p = score.max(1).values.softmax(0)
            # sum log p
            lp = p.new_zeros(1)
            # sum p
            sp = p.new_zeros(1)
            if action is None:
                if sample:
                    row = np.random.choice(range(len(score)), size=min(topk, len(score)), replace=False, p=p.detach().cpu().numpy())
                    act = []
                    for r in row:
                        c = np.random.choice(range(len(relay)), size=1, p=s_[r].detach().cpu().numpy())[0]
                        act.append([r, c])
                        lp = lp + torch.log(p[r]) - torch.log1p(-sp) + torch.log(s_[r, c])
                        sp = sp + p[r]
                    acts.append(torch.tensor(act, dtype=torch.long, device=custom.device))
                    log_probs.append(lp)
                else:
                    assert not self.training
                    with torch.no_grad():
                        row = p.topk(min(topk, len(p))).indices
                        if current_assign is not None:
                            ca = current_assign[i]
                            row_mask = s_.argmax(1) != ca
                            same_ind = torch.arange(len(row_mask))[~row_mask.cpu()]
                            if len(set(row.cpu().tolist()) - set(same_ind.tolist())) < least_action:
                                p_mask = p[row_mask]
                                s_ = s_[row_mask]
                                row = p_mask.topk(min(topk, len(p_mask))).indices
                        act = torch.tensor([[r, s_[r].argmax()] for r in row], dtype=torch.long, device=custom.device)
                        acts.append(act)
            else:
                for r, c in action[i]:
                    lp = lp + torch.log(p[r]) - torch.log1p(-sp) + torch.log(s_[r, c])
                    sp = sp + p[r]
                log_probs.append(lp)
        return action or acts, (torch.cat(log_probs) if sample else None), -(s_ * s_.log()).mean(), self.critic(state)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def polyak(m_out, m1, m2, k):
    params1 = m1.named_parameters()
    params2 = m2.named_parameters()
    out = dict(params2)
    for name1, param1 in params1:
        out[name1].data.copy_(k * param1.data + (1 - k) * out[name1].data)
    m_out.load_state_dict(out)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    run_name = args.name or time.strftime('PPO_%y%m%d_%H%M%S')
    print(f'Run: {run_name}')
    os.makedirs(f'log/{run_name}/code')
    os.makedirs(f'log/{run_name}/snapshot')
    os.makedirs(f'log/{run_name}/pt')
    for i in ['LKH', __file__, 'solver.py', 'utils.py']:
        shutil.copy(i, f'log/{run_name}/code/')
    with open(f'log/{run_name}/code/run', 'w') as f:
        f.write('#!/bin/sh\npython ' + ' '.join(sys.argv))

    json.dump(vars(args), open(f'log/{run_name}/args.json', 'w'), indent=4)

    writer = SummaryWriter(f"log/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    set_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu')

    agent = Agent(args.embed_dim, args.num_heads, args.num_layers, args.dist_hidden_dim, dropout=args.dropout,
                  share_base=args.share_base, use_mha=not args.no_mha, use_lstm=not args.no_lstm).to(device)
    if args.load:
        if '/' in args.load:
            path = args.load
        else:
            path = max(glob(f'log/{args.load}/pt/*.pt'), key=lambda x: int(re.findall(r'(\d+)', x.rsplit('/', 1)[1])[0]))
        print(f'Load model from {path}')
        param = torch.load(path, map_location=device)
        is_shared = any(i.startswith('base.') for i in param)
        if args.share_base == is_shared:
            agent.load_state_dict(param)
        else:
            old_agent = Agent(args.embed_dim, args.num_heads, args.num_layers, args.dist_hidden_dim, dropout=args.dropout, share_base=is_shared).to(device)
            old_agent.load_state_dict(param)
            if is_shared:
                print('Shared to not shared')
                agent.base_actor.load_state_dict(old_agent.base.state_dict())
                agent.base_critic.load_state_dict(old_agent.base.state_dict())
            else:
                print('Not shared to shared')
                polyak(agent.base, old_agent.base_actor, old_agent.base_critic, 0.5)
            agent.actor.load_state_dict(old_agent.actor.state_dict())
            agent.critic.load_state_dict(old_agent.critic.state_dict())
            del old_agent

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    k_cost = 1 / args.cost_norm
    k_penalty = 1 / args.penalty_norm
    args.workers = min(args.workers, args.num_instances)
    args.num_envs = min(args.num_envs, args.num_instances)
    args.num_parallel = min(args.num_parallel, args.num_instances)
    env = ParallelSolvers(parallel_map(lambda i: Solver(i, device=device, k_penalty=k_penalty,
                                                        k_cost=k_cost, k_cost_upper=args.cost_upper_coeff, use_incumbent_reward=args.incumbent_reward,
                                                        assign_init=args.assign_init, use_neg_cost=args.use_neg_cost,
                                                        always_eval_upper=args.always_eval_upper,
                                                        always_use_upper=args.always_use_upper,
                                                        lower_capacity=args.courier_capacity, courier_cost=args.courier_cost),
                          pickle.load(open(args.data, 'rb'))[:args.num_instances], args.workers), args.num_envs, args.num_parallel, args.episode, args.curriculum)

    obs = [None] * args.num_steps
    actions = [None] * args.num_steps
    log_probs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_time = time.time()

    next_obs = env.observe()
    next_done = torch.zeros(args.num_envs, device=device)
    tq = tqdm(range(args.total_timesteps), dynamic_ncols=True)
    tq.set_description(run_name)
    for global_step in tq:
        if global_step < 100 or global_step % 100 == 0:
            env.current_solvers[0].save_image(f'log/{run_name}/snapshot/{global_step}.png')

        for step in range(args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs, topk=args.step_action)
                values[step] = value.view(-1)
            actions[step] = action
            log_probs[step] = log_prob

            reward, next_obs, next_done, next_obs_ = env.step(action)
            rewards[step] = reward
            if next_obs_ is not None:
                next_obs = next_obs_

        costs = []
        for s in env.current_solvers:
            if np.sum(s.state.pl) + s.state.pu == 0:
                costs.append(np.sum(s.state.cl) + s.state.cu)
        if costs:
            writer.add_scalar("charts/cost", np.mean(costs), global_step)
            writer.add_scalar("charts/feasible", len(costs) / len(env.current_solvers), global_step)
        writer.add_scalar("charts/reward", reward.mean().cpu().item(), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                last_gae_lam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                    advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae_lam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * next_nonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = ObsIndexer(obs)
        b_log_probs = log_probs.reshape(-1)
        b_actions = [j for i in actions for j in i]
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], [b_actions[i] for i in mb_inds], topk=args.step_action)
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

        if (global_step + 1) % args.save_interval == 0:
            torch.save(agent.state_dict(), f'log/{run_name}/pt/{global_step+1}.pt')
    writer.close()


if __name__ == "__main__":
    main()
