# BLPDPTW

This repo contains the source code and datasets for our KDD'23 ADS track paper under review.

## Install

```bash
pip install -r requirements.txt
```

## Data generation

Generate 3 datasets with 10k instances and N = {100, 200, 400}.
* `data/100_10k.pkl`
* `data/200_10k.pkl`
* `data/400_10k.pkl`

```bash
python data_gen.py
```

## Model training

Train model on the datasets using the first 1k instances.

```bash
python train.py --cuda 0 --data data/100_10k.pkl --num-instances 1000
python train.py --cuda 0 --data data/200_10k.pkl --num-instances 1000
python train.py --cuda 0 --data data/400_10k.pkl --num-instances 1000
```

Ablation for MHA and LSTM modules

```bash
python train.py --cuda 0 --no-mha --data data/400_10k.pkl --num-instances 1000
python train.py --cuda 0 --no-lstm --data data/400_10k.pkl --num-instances 1000
python train.py --cuda 0 --no-mha --no-lstm --data data/400_10k.pkl --num-instances 1000
```

## Evaluation

Test the all the methods on the last 100 instances of each dataset.

**Non-RL methods**

```bash
python baseline.py -n {dataset_size} --method {method}
```

Here `dataset_size` can be {100, 200, 400}, `method` can be
* `near`: assign orders to their nearest contact station
* `kmeans`: use KMeans to cluster orders
* `gmm`: use Gaussian Mixture Model to cluster orders
* `sa`: Simulated Annealing
* `bb`: Blackbox optimization using RBFopt
* `bo`: Blackbox optimization using HEBO

**RL methods**

```bash
python eval_ppo.py --cuda 0 --name {exp_name}
```

Here `exp_name` is the name of the trained model. `eval_ppo.py` will construct the model and load weights based on the training log.