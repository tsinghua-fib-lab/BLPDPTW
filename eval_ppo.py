import argparse
import json
import os
import pickle
import re
import shutil
import sys
import time
from copy import deepcopy
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from train import Agent, ParallelSolvers, Solver, parallel_map, set_seed

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('-n', type=int, default=100)
parser.add_argument('--pt', type=int, default=0)
parser.add_argument('--step', type=int, default=48)
parser.add_argument('--step-action', type=int, default=4)

opt = parser.parse_args()


def make_object(d: dict):
    class Obj:
        def __init__(self, d):
            self.__dict__.update(d)
    return Obj(d)


name = opt.name
device = torch.device(f'cuda:{opt.cuda}')
path = 'log/' + name
args = make_object(json.load(open(f'{path}/args.json')))
args.episode = args.curriculum = 0
args.num_instances = abs(opt.n)
args.num_envs = args.num_parallel = min(abs(opt.n), 20)
if opt.n > 0:
    data = pickle.load(open(args.data, 'rb'))[:opt.n]
else:
    data = pickle.load(open(args.data, 'rb'))[opt.n:]
if args.step_action != opt.step_action:
    print(f'Step action changed from {args.step_action} to {opt.step_action}')
    args.step_action = opt.step_action

steps = opt.step
size = args.data.rsplit('/', 1)[-1].split('_', 1)[0].rsplit('x', 1)[-1]
if opt.pt == 0:
    pt, path = max(([int(re.findall(r'(\d+)', x.rsplit('/', 1)[1])[0]), x] for x in glob(f'{path}/pt/*.pt')), key=lambda x: x[0])
else:
    pt = opt.pt
    path = max(glob(f'{path}/pt/{opt.pt}*.pt'), key=lambda x: int(re.findall(r'(\d+)', x.rsplit('/', 1)[1])[0]))

output_folder = f'exp/{name}_{size}i{opt.n}_pt{pt}'
output_pkl = f'{output_folder}/s{opt.step}_a{opt.step_action}.pkl'
if os.path.exists(output_pkl):
    print('File already exists!')
    exit()

os.makedirs(output_folder, exist_ok=True)
code_path = output_folder + time.strftime('/code_%y%m%d_%H%M%S')
os.makedirs(code_path)
shutil.copy(__file__, code_path)
json.dump(vars(opt), open(f'{code_path}/opt.json', 'w'))
open(f'{code_path}/run', 'w').write('python ' + ' '.join(sys.argv))

set_seed(args.seed)
agent = Agent(args.embed_dim, args.num_heads, args.num_layers, args.dist_hidden_dim, dropout=args.dropout, share_base=args.share_base, use_mha=args.use_mha, use_lstm=args.use_lstm).to(device)
print(f'Load model from {path}')
agent.load_state_dict(torch.load(path, map_location=device))
agent.eval()
env = ParallelSolvers(parallel_map(lambda i: Solver(i, device=device, k_penalty=1, k_cost=1, use_incumbent_reward=args.incumbent_reward),
                                   data, args.workers), args.num_envs, args.num_parallel, args.episode, args.curriculum)
obs = env.observe()
fcs = [[[np.sum(s.state.pl), np.sum(s.state.cl), s.state.pu, s.state.cu]] for s in env.solvers]
sas = [[[deepcopy(s.state), deepcopy(s.assignment)]] for s in env.solvers]

with torch.no_grad():
    for global_step in tqdm(range(args.num_instances // args.num_envs * steps), dynamic_ncols=True):
        action, log_prob, _, value = agent.get_action_and_value(obs, topk=args.step_action, sample=False, current_assign=env.current_assignment())
        _, next_obs, done, next_obs_ = env.step(action)
        obs = next_obs if next_obs_ is None else next_obs_
        if env.i == 0:
            for i in env.solver_index:
                if env.solver_step_cnt[i] // env.n > len(fcs[i]):
                    s = env.solvers[i]
                    fcs[i].append([np.sum(s.state.pl), np.sum(s.state.cl), s.state.pu, s.state.cu])
                    sas[i].append([deepcopy(s.state), deepcopy(s.assignment)])

pickle.dump(fcs, open(output_pkl, 'wb'))
pickle.dump(sas, open(output_pkl[:-4] + '_sa.pkl', 'wb'))
