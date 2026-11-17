import sys
import subprocess
import json
import pickle
from pathlib import Path

root = Path.cwd()
source = root / "SunSight_Q3"
pin = "99588d1696c980075a19abf087c669ae5aa22e4e"
if not source.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "https://github.com/coopersigrist/SunSight.git", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", pin], check=True)
    subprocess.run(["git", "-C", str(source), "sparse-checkout", "add", "Simulation", "Models/Neat"], check=True)
if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != pin:
    raise RuntimeError("The source must be at the pinned commit.")
sys.path.insert(0, str(source))
sys.path.insert(0, str(root))


import copy
import itertools
import random
from pathlib import Path

import neat
import numpy as np
import pandas as pd

from q4_eva import FIELDS, OBJECTIVES, network_scores
from q5_softmax import softmax_allocation


def prepare_states(raw, state_names, n=2_000_000):
    states = sorted(state_names)
    if len(states) != 50 or len(set(states)) != 50:
        raise ValueError("Exactly 50 distinct state names are required.")
    data = raw.loc[raw["state_name"].isin(states)].sort_values("region_name").reset_index(drop=True).copy()
    required = FIELDS + ["count_qualified", "existing_installs_count"]
    if set(data["state_name"]) != set(states) or data["region_name"].duplicated().any():
        raise ValueError("Every state needs data and ZIP identifiers must be unique.")
    if not np.isfinite(data[required].to_numpy()).all() or not data["count_qualified"].gt(0).all():
        raise ValueError("Required attributes must be finite and capacities positive.")
    capacity = np.floor(data["count_qualified"]).to_numpy(dtype=np.int64)
    if n <= 0 or n > capacity.sum():
        raise ValueError("The installation budget must be positive and feasible.")
    groups = pd.Categorical(data["state_name"], categories=states).codes
    if (data["existing_installs_count"] < 0).any() or (np.bincount(groups, weights=data["existing_installs_count"], minlength=50) <= 0).any():
        raise ValueError("Every state needs positive existing installations to define no-new-allocation equity.")
    signs = []
    national_signs = []
    for field in ["black_prop", "Median_income"]:
        medians = data.groupby("state_name")[field].transform("median")
        high = data[field].ge(medians).to_numpy()
        if any(high[groups == s].all() or not high[groups == s].any() for s in range(50)):
            raise ValueError("Both demographic groups must be present in every state.")
        signs.append(np.where(high, 1.0, -1.0))
        national_signs.append(np.where(data[field].ge(data[field].median()), 1.0, -1.0))
    values = data[FIELDS].to_numpy()
    spread = np.ptp(values, axis=0)
    inputs = np.column_stack([np.ones(len(data)), (values - values.min(axis=0)) / np.where(spread > 0, spread, 1)])
    experiment = {"data": data, "states": states, "groups": groups, "capacity": capacity, "inputs": inputs, "existing": data["existing_installs_count"].to_numpy(), "carbon": data[FIELDS[1]].to_numpy(), "energy": data[FIELDS[2]].to_numpy(), "signs": np.array(signs), "national_signs": np.array(national_signs), "n": int(n)}
    reference = n * capacity / capacity.sum()
    experiment["reference"] = reference
    experiment["state_reference"] = state_values(experiment, reference)
    experiment["national_reference"] = national_values(experiment, reference)
    if (experiment["state_reference"] <= 0).any() or (experiment["national_reference"] <= 0).any():
        raise ValueError("Reference objectives must be positive; no arbitrary denominator smoothing is used.")
    return experiment


def state_values(experiment, allocation):
    groups = experiment["groups"]
    installed = experiment["existing"] + allocation
    totals = np.bincount(groups, weights=installed, minlength=50)
    carbon = np.bincount(groups, weights=allocation * experiment["carbon"], minlength=50)
    energy = np.bincount(groups, weights=allocation * experiment["energy"], minlength=50)
    equity = [1 - np.abs(np.bincount(groups, weights=installed * sign, minlength=50)) / totals for sign in experiment["signs"]]
    return np.column_stack([carbon, energy, *equity])


def national_values(experiment, allocation):
    installed = experiment["existing"] + allocation
    equity = 1 - np.abs(experiment["national_signs"] @ installed) / installed.sum()
    return np.array([allocation @ experiment["carbon"], allocation @ experiment["energy"], *equity])


def candidate_fitness(experiment, network, temperature, placement_seeds):
    scores = network_scores(network, experiment["inputs"])
    local = np.zeros((50, 4))
    national = np.zeros(4)
    for seed in placement_seeds:
        allocation = softmax_allocation(scores, experiment["capacity"], experiment["n"], temperature, np.random.default_rng(seed))
        local += state_values(experiment, allocation)
        national += national_values(experiment, allocation)
    return local / len(placement_seeds) / experiment["state_reference"], national / len(placement_seeds) / experiment["national_reference"]


def sample_cases(rng, states_per_generation=10):
    if not isinstance(states_per_generation, (int, np.integer)) or not 1 <= states_per_generation <= 50:
        raise ValueError("Sample between 1 and 50 states.")
    states = rng.choice(50, size=states_per_generation, replace=False)
    return (4 * states[:, None] + np.arange(4)).ravel()


def epsilon_survivors(fitness, cases, count, rng):
    fitness = np.asarray(fitness, dtype=float)
    cases = np.asarray(cases, dtype=int)
    if fitness.ndim != 2 or not np.isfinite(fitness).all() or not 1 <= count <= len(fitness):
        raise ValueError("A finite candidate-by-objective matrix and feasible survivor count are required.")
    if len(cases) == 0 or len(np.unique(cases)) != len(cases) or (cases < 0).any() or (cases >= fitness.shape[1]).any():
        raise ValueError("Cases must be distinct valid objective indices.")
    median = np.median(fitness, axis=0)
    epsilon = np.median(np.abs(fitness - median), axis=0)
    available = np.arange(len(fitness))
    selected = []
    events = []
    for slot in range(count):
        pool = available.copy()
        order = rng.permutation(cases)
        for position, case in enumerate(order):
            if len(pool) == 1:
                break
            before = len(pool)
            best = fitness[pool, case].max()
            pool = pool[fitness[pool, case] >= best - epsilon[case]]
            events.append([slot, position, int(case), before, len(pool)])
        winner = int(rng.choice(pool))
        selected.append(winner)
        available = available[available != winner]
    return np.array(selected), events


def train_state_eva(experiment, config_path, seed, mode="state_subset", generations=30, temperature=0.01, placement_seeds=(10, 11, 12, 13), states_per_generation=10, checkpoint="cpr"):
    if checkpoint not in ["cpr", "log"]:
        raise ValueError("Checkpoint scoring must be cpr or log.")
    if mode not in ["national", "state_mean", "state_all", "state_subset"]:
        raise ValueError("Unknown selection mode.")
    random.seed(seed)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
    if len(config.genome_config.input_keys) != experiment["inputs"].shape[1]:
        raise ValueError("The network input size does not match the normalized data.")
    selection_rng = np.random.default_rng(100_000 + seed)
    subset_rng = np.random.default_rng(200_000 + seed)
    population = []
    next_id = 0
    for _ in range(30):
        genome = neat.DefaultGenome(next_id)
        next_id += 1
        genome.configure_new(config.genome_config)
        population.append(genome)
    best_genome = None
    best_score = -np.inf
    history = []
    subset_log = []
    event_log = []
    for generation in range(generations):
        local_fitness = []
        national_fitness = []
        for genome in population:
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            local, national = candidate_fitness(experiment, network, temperature, placement_seeds)
            local_fitness.append(local.ravel())
            national_fitness.append(national)
        local_fitness = np.array(local_fitness)
        national_fitness = np.array(national_fitness)
        fitness = national_fitness if mode == "national" else local_fitness.reshape(30, 50, 4).mean(axis=1) if mode == "state_mean" else local_fitness
        cpr = fitness.sum(axis=1) if mode == "national" else fitness.mean(axis=1) * 4
        if checkpoint == "log":
            cpr = 4 * np.log1p(fitness).mean(axis=1) / np.log(2)
        for genome, score in zip(population, cpr):
            genome.fitness = float(score)
        cases = np.arange(4) if mode in ["national", "state_mean"] else np.arange(200) if mode == "state_all" else sample_cases(subset_rng, states_per_generation)
        survivors, events = epsilon_survivors(fitness, cases, 9, selection_rng)
        for case in cases:
            subset_log.append([generation, int(case)])
        event_log.extend([[generation, *event] for event in events])
        winner = int(survivors[np.argmax(cpr[survivors])])
        history.append([generation, float(cpr[winner]), *national_fitness[winner], *local_fitness[winner].reshape(50, 4).mean(axis=0)])
        if cpr[winner] > best_score:
            best_score = float(cpr[winner])
            best_genome = copy.deepcopy(population[winner])
        if generation + 1 < generations:
            selected = [population[i] for i in survivors]
            pairs = list(itertools.combinations(selected, 2))
            random.shuffle(pairs)
            new_population = [copy.deepcopy(population[winner])]
            for first, second in pairs[:29]:
                child = neat.DefaultGenome(next_id)
                next_id += 1
                child.configure_crossover(first, second, config.genome_config)
                child.mutate(config.genome_config)
                new_population.append(child)
            population = new_population
    history = pd.DataFrame(history, columns=["Generation", "Training checkpoint score", *["National " + x for x in OBJECTIVES], *["State mean " + x for x in OBJECTIVES]])
    subsets = pd.DataFrame(subset_log, columns=["Generation", "Case"])
    events = pd.DataFrame(event_log, columns=["Generation", "Slot", "Position", "Case", "Before", "After"])
    return neat.nn.FeedForwardNetwork.create(best_genome, config), history, subsets, events


def evaluate_state_model(experiment, network, temperature=0.01, deployment_seeds=range(20_000, 20_100)):
    scores = network_scores(network, experiment["inputs"])
    rows = []
    state_rows = []
    for seed in deployment_seeds:
        allocation = softmax_allocation(scores, experiment["capacity"], experiment["n"], temperature, np.random.default_rng(seed))
        local = state_values(experiment, allocation) / experiment["state_reference"]
        national = national_values(experiment, allocation) / experiment["national_reference"]
        allocated = np.bincount(experiment["groups"], weights=allocation, minlength=50)
        rows.append([seed, *national, *local.mean(axis=0), int((allocated > 0).sum()), int((allocation > 0).sum())])
        state_rows.extend([[seed, state, allocated[s], *local[s]] for s, state in enumerate(experiment["states"])])
    columns = ["Deployment seed", *["National " + x for x in OBJECTIVES], *["State mean " + x for x in OBJECTIVES], "States allocated", "ZIPs allocated"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(state_rows, columns=["Deployment seed", "State", "Allocation", *OBJECTIVES])


state_names = json.loads((root / "states.json").read_text())
experiment = prepare_states(pd.read_csv(root / "data_by_zip.csv"), state_names)
config_path = source / "Models" / "Neat" / "neat-config"
print("ZIPs:", len(experiment["data"]), "States:", len(experiment["states"]), "Objectives:", experiment["state_reference"].size)


retrain = False
models = {}
for label in ["national", "state_mean", "state_all", "state_subset", "state_subset_log"]:
    for seed in range(5):
        path = root / f"{label}_seed_{seed}.pkl"
        if retrain:
            mode = "state_subset" if label == "state_subset_log" else label
            checkpoint = "log" if label == "state_subset_log" else "cpr"
            network, history, subsets, events = train_state_eva(experiment, config_path, seed, mode, checkpoint=checkpoint)
            path.write_bytes(pickle.dumps(network))
            history.to_csv(root / f"{label}_history_{seed}.csv", index=False)
            subsets.to_csv(root / f"{label}_subsets_{seed}.csv", index=False)
            events.to_csv(root / f"{label}_events_{seed}.csv", index=False)
        else:
            network = pickle.loads(path.read_bytes())
        models[(label, seed)] = network


for (label, seed), network in models.items():
    national, local = evaluate_state_model(experiment, network, deployment_seeds=range(40_000, 40_100))
    national.to_csv(root / f"{label}_national_test_{seed}.csv", index=False)
    local.to_csv(root / f"{label}_state_test_{seed}.csv", index=False)


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q4_eva import OBJECTIVES


def summarize_results(root):
    root = Path(root)
    endpoint = []
    state = []
    coverage = []
    for mode in ["national", "state_mean", "state_all", "state_subset", "state_subset_log"]:
        for seed in range(5):
            national = pd.read_csv(root / f"{mode}_national_test_{seed}.csv")
            local = pd.read_csv(root / f"{mode}_state_test_{seed}.csv")
            average = local.groupby("State")[OBJECTIVES].mean()
            metrics = national.drop(columns="Deployment seed").mean().to_dict()
            metrics.update({"Mode": mode, "Evolution seed": seed})
            for objective in OBJECTIVES:
                metrics["State median " + objective] = average[objective].median()
                metrics["State minimum " + objective] = average[objective].min()
                metrics["State Q10 " + objective] = average[objective].quantile(0.1)
                metrics["States below reference " + objective] = int((average[objective] < 1).sum())
            metrics["States meeting all four in held-out means"] = int(average.ge(1).all(axis=1).sum())
            metrics["State CPR"] = average.to_numpy().mean() * 4
            metrics["National CPR"] = sum(metrics["National " + name] for name in OBJECTIVES)
            endpoint.append(metrics)
            average["Mode"] = mode
            average["Evolution seed"] = seed
            state.append(average.reset_index())
            if mode in ["state_all", "state_subset", "state_subset_log"]:
                subsets = pd.read_csv(root / f"{mode}_subsets_{seed}.csv")
                events = pd.read_csv(root / f"{mode}_events_{seed}.csv")
                counts = subsets.assign(StateIndex=subsets["Case"] // 4).groupby("StateIndex").size().reindex(range(50), fill_value=0) / 4
                effective = events.loc[events["After"] < events["Before"]].assign(StateIndex=lambda x: x["Case"] // 4).groupby("StateIndex").size().reindex(range(50), fill_value=0)
                depths = events.groupby(["Generation", "Slot"]).size()
                coverage.append({"Mode": mode, "Evolution seed": seed, "States ever sampled": int((counts > 0).sum()), "Minimum sampled generations": counts.min(), "Maximum sampled generations": counts.max(), "Mean cases examined per slot": depths.mean(), "Fraction examined cases reducing pool": (events["After"] < events["Before"]).mean(), "States ever reducing a pool": int((effective > 0).sum())})
    endpoint = pd.DataFrame(endpoint)
    state = pd.concat(state, ignore_index=True)
    coverage = pd.DataFrame(coverage)
    endpoint.to_csv(root / "per_run_summary.csv", index=False)
    state.to_csv(root / "state_heldout_means.csv", index=False)
    coverage.to_csv(root / "selection_diagnostics.csv", index=False)
    metrics = endpoint.select_dtypes(include=np.number).columns.drop("Evolution seed")
    endpoint.groupby("Mode")[metrics].mean().to_csv(root / "five_seed_means.csv")
    endpoint.groupby("Mode")[metrics].std().to_csv(root / "five_seed_sd.csv")
    return endpoint, state, coverage


def plot_results(root, endpoint, state):
    root = Path(root)
    labels = {"national": "National 4", "state_mean": "State means 4", "state_all": "State 200", "state_subset": "State subset 40", "state_subset_log": "Subset 40 + log score"}
    colors = dict(zip(labels, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, objective in zip(axes, ["Carbon", "Energy"]):
        for mode in labels:
            points = endpoint.loc[endpoint["Mode"] == mode]
            ax.scatter(points["National " + objective], points["State mean " + objective], label=labels[mode], color=colors[mode], s=35)
        ax.axhline(1, color="gray", linestyle=":")
        ax.axvline(1, color="gray", linestyle=":")
        ax.set(title=objective, xlabel="National ratio to capacity reference", ylabel="Unweighted mean of 50 state ratios")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("National performance and the average state can move in opposite directions | five training seeds")
    fig.savefig(root / "national_vs_state.png", dpi=180)
    plt.show()
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, objective in zip(axes.flat, OBJECTIVES):
        for mode in labels:
            values = state.loc[(state["Mode"] == mode) & (state["Evolution seed"] == 0), objective].sort_values().to_numpy()
            ax.plot(np.arange(1, 51), values, label=labels[mode], color=colors[mode])
        ax.axhline(1, color="gray", linestyle=":")
        ax.set(title=objective, xlabel="State rank within each policy, worst to best", ylabel="Held-out mean state ratio")
        ax.grid(alpha=0.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Primary training seed 0 | each curve sorts its own 50 states | 100 deployment draws")
    fig.savefig(root / "state_distribution.png", dpi=180)
    plt.show()
    plt.close(fig)


endpoint, state, coverage = summarize_results(root)
columns = ["National " + name for name in OBJECTIVES] + ["State mean " + name for name in OBJECTIVES] + ["States meeting all four in held-out means"]
print(endpoint.groupby("Mode")[columns].mean().round(4).to_string())
print(endpoint.groupby("Mode")[columns].std().round(4).to_string())
print(coverage.to_string(index=False))
plot_results(root, endpoint, state)


assert len(experiment["states"]) == 50 and experiment["state_reference"].shape == (50, 4)
assert np.allclose(state_values(experiment, experiment["reference"]) / experiment["state_reference"], 1)
scores = network_scores(models[("state_subset_log", 0)], experiment["inputs"])
allocation = softmax_allocation(scores, experiment["capacity"], experiment["n"], 0.01, np.random.default_rng(12345))
assert allocation.sum() == experiment["n"] and (allocation <= experiment["capacity"]).all()
assert np.allclose(state_values(experiment, allocation)[:, :2].sum(axis=0), national_values(experiment, allocation)[:2])
cases = sample_cases(np.random.default_rng(5), 10)
assert len(cases) == 40 and len(set(cases // 4)) == 10
print("Objective dimensions, reference identity, exact budget, capacity, physical additivity and subset checks passed.")

