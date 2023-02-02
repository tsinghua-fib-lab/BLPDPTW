import pickle

import igraph
import numpy as np
from perlin_numpy import generate_fractal_noise_2d
from tqdm import tqdm

from constants import *


def perlin(shape=256, res=8, octaves=5):
    m = generate_fractal_noise_2d((shape,) * 2, (res,) * 2, octaves)
    m = (m - m.min()) / (m.max() - m.min())
    return m


def _grid(x, nx):
    return np.square(np.clip(np.abs(x * nx - np.round(x * nx)) * 10 - .05, 0, 1))


def grid(nx, ny, res=256):
    x, y = np.meshgrid(*(np.linspace(0, 1, res),) * 2)
    z = np.minimum(_grid(x, nx), _grid(y, ny))
    return z


def prob_mask(xy, prob):
    px, py = np.floor(xy * np.array(prob.shape)).T.astype(int)
    mask = np.random.rand(len(xy)) < prob[py, px]
    return xy[mask, :]


def generate(path, instance_size, instance_count, nx=2, ny=2, seed=43):
    edges = []
    relay_id = {j: i for i, j in enumerate((x, y, i) for y in range(ny) for x in range(nx) for i in range(4))}
    inter_id = {j: i + len(relay_id) for i, j in enumerate((x, y) for y in range(ny + 1) for x in range(nx + 1))}
    for x in range(nx):
        for y in range(ny):
            edges.append((inter_id[x, y], relay_id[x, y, 0]))
            edges.append((relay_id[x, y, 0], inter_id[x + 1, y]))
            edges.append((inter_id[x + 1, y], relay_id[x, y, 1]))
            edges.append((relay_id[x, y, 1], inter_id[x + 1, y + 1]))
            edges.append((inter_id[x + 1, y + 1], relay_id[x, y, 2]))
            edges.append((relay_id[x, y, 2], inter_id[x, y + 1]))
            edges.append((inter_id[x, y + 1], relay_id[x, y, 3]))
            edges.append((relay_id[x, y, 3], inter_id[x, y]))
    G = igraph.Graph(edges, directed=True)
    G.es['length'] = el = sum([[200, 100]] * 4, []) * (len(edges) // 8)
    ps = [inter_id[nx // 2, ny // 2]] + list(relay_id.values())
    relay_dist = np.zeros((len(ps),) * 2)
    for i, pi in enumerate(ps):
        for j, pj in enumerate(ps):
            if i != j:
                relay_dist[i, j] = sum(el[k] for k in G.get_shortest_paths(pi, pj, weights='length', output='epath')[0])
    relay_pos = np.array([[nx // 2 * 3, ny // 2 * 3]] + sum([[[x * 3 + 2, y * 3 + 3], [x * 3 + 3, y * 3 + 1], [x * 3 + 1, y * 3], [x * 3, y * 3 + 2], ]
                         for x in range(nx) for y in range(ny)], [])) * 100
    np.random.seed(seed)
    out = []
    for _ in tqdm(range(instance_count)):
        prob = generate_fractal_noise_2d((256,) * 2, (4,) * 2, 4)
        prob = (prob - prob.min()) / (prob.max() - prob.min())
        prob **= 4
        p = np.random.rand(10000, 2)
        p = prob_mask(p, prob)
        p = prob_mask(p, grid(2, 2))
        p = p[:nx * ny * instance_size // 4]
        ix, iy = np.floor(p * np.array([nx, ny])).T.astype(int)
        ixy = iy + ix * ny
        p = p * np.array([nx, ny]) * 300
        t = np.random.randint(0, SPAN - TW - SERVICE, p.shape[:1])
        tw = np.array([t, t + TW, [SERVICE] * len(t)]).T
        d = 1 + np.clip(np.random.randn(len(p)) * .3, -.5, .5)
        mask = np.random.rand(len(p)) < .1
        demand = np.zeros((len(p), 2))
        demand[mask, 0] = d[mask]
        demand[~mask, 1] = d[~mask]
        out.append({
            'nx': nx,
            'ny': ny,
            'relay_pos': relay_pos,
            'relay_dist': relay_dist,
            'custom_pos': p,
            'custom_region': ixy,
            'custom_demand': np.hstack([demand, tw])
        })
    pickle.dump(out, open(path, 'wb'))


def main():
    for size in [100, 200, 400]:
        generate(f'data/{size}_10k.pkl', size, 10000)
