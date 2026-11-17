import copy
import itertools
import random

import neat
import numpy as np
import pandas as pd

from q4_eva import OBJECTIVES, baseline_allocation, evaluate, network_scores, rank_allocation


def softmax_probabilities(scores, temperature):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite one-dimensional array.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    probabilities = np.exp((scores - scores.max()) / temperature)
    return probabilities / probabilities.sum()


def softmax_allocation(scores, capacities, n, temperature, rng):
    scores = np.asarray(scores, dtype=float)
    capacities = np.asarray(capacities)
    if scores.ndim != 1 or capacities.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Scores and capacities must be matching finite one-dimensional arrays.")
    if not np.isfinite(capacities).all() or (capacities < 0).any() or (capacities != np.floor(capacities)).any():
        raise ValueError("Capacities must be nonnegative integers.")
    if not isinstance(n, (int, np.integer)) or n < 0 or n > capacities.sum():
        raise ValueError("n must be a feasible nonnegative integer.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    capacities = capacities.astype(np.int64)
    allocation = np.zeros(len(scores), dtype=np.int64)
    remaining = int(n)
    while remaining:
        active = np.flatnonzero(allocation < capacities)
        probabilities = softmax_probabilities(scores[active], temperature)
        proposals = rng.multinomial(remaining, probabilities)
        accepted = np.minimum(proposals, capacities[active] - allocation[active])
        allocation[active] += accepted
        remaining -= int(accepted.sum())
    return allocation


def mean_objectives(experiment, scores, n, temperature, seeds):
    values = []
    for seed in seeds:
        allocation = softmax_allocation(scores, experiment["capacity"], n, temperature, np.random.default_rng(seed))
        values.append(evaluate(experiment, allocation))
    return np.mean(values, axis=0)


def train_policy(experiment, config_path, seed, temperature=None, training_seeds=(10, 11, 12, 13), generations=6, n=2_000_000, selectivity=None):
    random.seed(seed)
    np.random.seed(seed)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
    baseline = evaluate(experiment, baseline_allocation(experiment, n))
    selectivity = np.ones(4) if selectivity is None else np.asarray(selectivity, dtype=float)
    population = []
    next_id = 0
    for _ in range(30):
        genome = neat.DefaultGenome(next_id)
        next_id += 1
        genome.configure_new(config.genome_config)
        population.append(genome)
    best_genome = None
    best_cpr = -np.inf
    history = []
    for generation in range(generations):
        ratios = []
        for genome in population:
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            scores = network_scores(network, experiment["inputs"])
            if temperature is None:
                values = evaluate(experiment, rank_allocation(scores, experiment["capacity"], n))
            else:
                values = mean_objectives(experiment, scores, n, temperature, training_seeds)
            ratios.append(values / baseline)
        ratios = np.array(ratios)
        cpr = ratios.sum(axis=1)
        for genome, fitness in zip(population, cpr):
            genome.fitness = float(fitness)
        survivors = np.arange(len(population))
        for objective in np.random.permutation(4):
            survivors = survivors[np.argsort(-ratios[survivors, objective], kind="stable")]
            keep = int(np.ceil(len(survivors) * 0.3 ** (selectivity[objective] / selectivity.sum())))
            survivors = survivors[:keep]
        survivors = survivors[np.argsort(-cpr[survivors], kind="stable")][:9]
        winner = int(survivors[0])
        history.append([generation, *ratios[winner], cpr[winner]])
        if cpr[winner] > best_cpr:
            best_cpr = float(cpr[winner])
            best_genome = copy.deepcopy(population[winner])
        if generation + 1 < generations:
            selected = [population[i] for i in survivors]
            pairs = list(itertools.combinations(selected, 2))
            random.shuffle(pairs)
            new_population = [copy.deepcopy(selected[0])]
            for first, second in pairs[:29]:
                child = neat.DefaultGenome(next_id)
                next_id += 1
                child.configure_crossover(first, second, config.genome_config)
                child.mutate(config.genome_config)
                new_population.append(child)
            population = new_population
    network = neat.nn.FeedForwardNetwork.create(best_genome, config)
    return network, pd.DataFrame(history, columns=["Generation", *OBJECTIVES, "CPR"])


def train_softmax_with_tuning(experiment, config_path, seed=0, temperature=0.01, training_seeds=(10, 11, 12, 13), max_rounds=6):
    selectivity = np.ones(4)
    baseline = evaluate(experiment, baseline_allocation(experiment, 2_000_000))
    best_network = None
    best_history = None
    best_cpr = -np.inf
    rows = []
    columns = ["Stage", *["Selectivity: " + name for name in OBJECTIVES], *OBJECTIVES]
    for stage in range(max_rounds):
        network, history = train_policy(experiment, config_path, seed + 1000 * stage, temperature, training_seeds, selectivity=selectivity)
        scores = network_scores(network, experiment["inputs"])
        ratios = mean_objectives(experiment, scores, 2_000_000, temperature, training_seeds) / baseline
        rows.append([stage, *selectivity, *ratios])
        if ratios.sum() > best_cpr:
            best_network = network
            best_history = history
            best_cpr = float(ratios.sum())
        if (ratios >= 1).all():
            return network, pd.DataFrame(rows, columns=columns), history
        selectivity = np.where(ratios < 1, selectivity / np.maximum(ratios, 1e-9), selectivity)
    return best_network, pd.DataFrame(rows, columns=columns), best_history


def softmax_path(scores, capacities, budgets, temperature, rng):
    budgets = np.asarray(budgets)
    if budgets.ndim != 1 or (budgets < 0).any() or (budgets != np.floor(budgets)).any() or (np.diff(budgets) < 0).any():
        raise ValueError("Budgets must be ordered nonnegative integers.")
    if len(budgets) and budgets[-1] > np.sum(capacities):
        raise ValueError("The final budget exceeds capacity.")
    allocation = np.zeros(len(scores), dtype=np.int64)
    path = []
    previous = 0
    for n in budgets:
        allocation += softmax_allocation(scores, capacities - allocation, int(n - previous), temperature, rng)
        path.append(allocation.copy())
        previous = int(n)
    return path


def allocation_diagnostics(allocation):
    shares = allocation[allocation > 0] / allocation.sum()
    return {"ZIPs used": int(len(shares)), "Effective allocated ZIPs": float(np.exp(-np.sum(shares * np.log(shares)))), "Largest ZIP share (%)": float(100 * shares.max())}


def evaluate_policy(experiment, network, temperature, seeds, n=2_000_000):
    scores = network_scores(network, experiment["inputs"])
    baseline = evaluate(experiment, baseline_allocation(experiment, n))
    rows = []
    for seed in seeds:
        allocation = rank_allocation(scores, experiment["capacity"], n) if temperature is None else softmax_allocation(scores, experiment["capacity"], n, temperature, np.random.default_rng(seed))
        ratios = evaluate(experiment, allocation) / baseline
        rows.append({"Deployment seed": int(seed), **dict(zip(OBJECTIVES, ratios)), **allocation_diagnostics(allocation)})
    return pd.DataFrame(rows)
