import sys
import subprocess
from pathlib import Path

root = Path.cwd()
source = root / "SunSight_Q3"
pin = "99588d1696c980075a19abf087c669ae5aa22e4e"
if not source.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "https://github.com/coopersigrist/SunSight.git", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", pin], check=True)
    subprocess.run(["git", "-C", str(source), "sparse-checkout", "add", "Simulation", "Models/Neat"], check=True)
if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != pin:
    raise RuntimeError("SunSight_Q3 must be at the pinned commit.")
sys.path.insert(0, str(source))
sys.path.insert(0, str(root))


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from q4_eva import prepare_experiment

experiment = prepare_experiment(pd.read_csv(root / "data_by_zip.csv"), pd.read_csv(root / "Q4_EIA_March_2024_2025.csv"))
config_path = source / "Models" / "Neat" / "neat-config"
temperature = 0.01
evolution_seeds = [0, 1, 2]
training_seeds = (10, 11, 12, 13)
test_seeds = range(20_000, 20_100)
q4_models = {seed: pickle.loads((root / f"eva_seed_{seed}.pkl").read_bytes()) for seed in evolution_seeds}


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


softmax_models = {}
training_rows = []
for seed in evolution_seeds:
    network, tuning, history = train_softmax_with_tuning(experiment, config_path, seed, temperature, training_seeds, max_rounds=6)
    softmax_models[seed] = network
    (root / f"softmax_seed_{seed}.pkl").write_bytes(pickle.dumps(network))
    tuning.to_csv(root / f"tuning_seed_{seed}.csv", index=False)
    history.to_csv(root / f"training_seed_{seed}.csv", index=False)
    passed = tuning[OBJECTIVES].ge(1).all(axis=1)
    selected = int(tuning.index[passed][0]) if passed.any() else int(tuning[OBJECTIVES].sum(axis=1).idxmax())
    training_rows.append({"Evolution seed": seed, "Tuning rounds": len(tuning), "Selected stage": selected, "Training criterion met": bool(passed.any())})
training_status = pd.DataFrame(training_rows)
training_status.to_csv(root / "training_status.csv", index=False)
print(training_status.to_string(index=False))


rows = []
for seed in evolution_seeds:
    policies = [("Q4 deterministic", q4_models[seed], None), ("Q4 network + softmax", q4_models[seed], temperature), ("Retrained softmax EVA", softmax_models[seed], temperature)]
    for label, network, policy_temperature in policies:
        samples = evaluate_policy(experiment, network, policy_temperature, test_seeds if policy_temperature is not None else [20_000])
        samples["Evolution seed"] = seed
        samples["Policy"] = label
        rows.append(samples)
endpoint_samples = pd.concat(rows, ignore_index=True)
metrics = OBJECTIVES + ["ZIPs used", "Effective allocated ZIPs", "Largest ZIP share (%)"]
means = endpoint_samples.groupby(["Evolution seed", "Policy"])[metrics].mean()
sds = endpoint_samples.groupby(["Evolution seed", "Policy"])[metrics].std().fillna(0)
endpoint_samples.to_csv(root / "endpoint_samples.csv", index=False)
means.to_csv(root / "endpoint_means.csv")
sds.to_csv(root / "endpoint_deployment_sd.csv")
print(means.round(6).to_string())
print(sds[OBJECTIVES].round(6).to_string())


rows = []
for t in [1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]:
    samples = evaluate_policy(experiment, q4_models[0], t, test_seeds)
    samples["Temperature"] = t
    rows.append(samples)
temperature_samples = pd.concat(rows, ignore_index=True)
temperature_samples.to_csv(root / "temperature_samples.csv", index=False)
summary = temperature_samples.groupby("Temperature")[OBJECTIVES + ["ZIPs used"]].mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
for objective in OBJECTIVES:
    axes[0].semilogx(summary.index, summary[objective], marker="o", label=objective)
axes[0].axhline(1, color="black", linestyle=":")
axes[0].set(xlabel="Temperature", ylabel="Mean ratio to reconstructed status quo")
axes[0].legend()
axes[1].semilogx(summary.index, summary["ZIPs used"], marker="o")
axes[1].axhline(289, color="gray", linestyle="--")
axes[1].set(xlabel="Temperature", ylabel="Mean ZIPs receiving units")
fig.savefig(root / "temperature_sensitivity.png", dpi=180)
plt.show()
plt.close(fig)


budgets = np.arange(100_000, 2_000_001, 100_000)
scores = network_scores(softmax_models[0], experiment["inputs"])
baselines = {int(n): evaluate(experiment, baseline_allocation(experiment, int(n))) for n in budgets}
rows = []
for seed in range(30_000, 30_100):
    path = softmax_path(scores, experiment["capacity"], budgets, temperature, np.random.default_rng(seed))
    previous = np.zeros(len(scores), dtype=np.int64)
    for n, allocation in zip(budgets, path):
        assert allocation.sum() == n and (allocation <= experiment["capacity"]).all() and (allocation >= previous).all()
        previous = allocation
        rows.append([seed, int(n), *evaluate(experiment, allocation) / baselines[int(n)]])
path_samples = pd.DataFrame(rows, columns=["Deployment seed", "Installations", *OBJECTIVES])
path_samples.to_csv(root / "nested_path_samples.csv", index=False)
path_means = path_samples.groupby("Installations")[OBJECTIVES].mean()
path_sd = path_samples.groupby("Installations")[OBJECTIVES].std()
greedy = np.array([evaluate(experiment, rank_allocation(network_scores(q4_models[0], experiment["inputs"]), experiment["capacity"], int(n))) / baselines[int(n)] for n in budgets])
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for k, (ax, objective) in enumerate(zip(axes.flat, OBJECTIVES)):
    x = budgets / 1e6
    y = path_means[objective].to_numpy()
    sd = path_sd[objective].to_numpy()
    ax.plot(x, greedy[:, k], linestyle="--", label="Q4 greedy")
    ax.plot(x, y, label="Retrained softmax")
    ax.fill_between(x, y - sd, y + sd, alpha=0.2)
    ax.axhline(1, color="black", linestyle=":")
    ax.set(title=objective, xlabel="Additional installation units (millions)", ylabel="Ratio to reconstructed status quo")
axes.flat[0].legend()
fig.suptitle("Primary seed 0: mean and one deployment SD across 100 nested paths")
fig.savefig(root / "nested_deployment.png", dpi=180)
plt.show()
plt.close(fig)


scores = network_scores(softmax_models[0], experiment["inputs"])
a = softmax_allocation(scores, experiment["capacity"], 2_000_000, temperature, np.random.default_rng(12345))
b = softmax_allocation(scores, experiment["capacity"], 2_000_000, temperature, np.random.default_rng(12345))
assert np.array_equal(a, b)
assert a.sum() == 2_000_000 and (a >= 0).all() and (a <= experiment["capacity"]).all()
assert np.issubdtype(a.dtype, np.integer)
assert np.allclose(scores, np.array([softmax_models[0].activate(x)[0] for x in experiment["inputs"]]), atol=1e-14, rtol=0)
print("Repeatability, integer counts, exact budget, capacities and inference agreement passed; nested paths checked above.")

