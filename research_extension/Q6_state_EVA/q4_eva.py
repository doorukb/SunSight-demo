import copy
import itertools
import random
from pathlib import Path

import neat
import numpy as np
import pandas as pd

from Simulation.projections_util import Objective, Projection


FIELDS = ["Median_income", "carbon_offset_kg_per_panel", "yearly_sunlight_kwh_kw_threshold_avg", "panel_utilization", "black_prop"]
OBJECTIVES = ["Carbon", "Energy", "Racial equity", "Income equity"]


def linear_objective(frame, allocation, column):
    return float(allocation @ frame[column].to_numpy())


def equity_objective(frame, allocation, field):
    high = frame[field].ge(frame[field].median()).to_numpy()
    installed = allocation + frame["existing_installs_count"].to_numpy()
    return 1 - abs(installed[high].sum() - installed[~high].sum()) / installed.sum()


def prepare_experiment(raw, eia):
    complete = eia.dropna(subset=["March2025", "March2024"]).set_index("State")
    data = raw.loc[raw["state_name"].isin(complete.index)].sort_values("region_name").reset_index(drop=True).copy()
    required = FIELDS + ["count_qualified", "existing_installs_count"]
    if not np.isfinite(data[required].to_numpy()).all() or not data["count_qualified"].gt(0).all():
        raise ValueError("Required EVA fields must be finite, with positive capacities.")
    if data["region_name"].duplicated().any():
        raise ValueError("ZIP identifiers must be unique.")
    values = data[FIELDS].to_numpy()
    spread = values.max(axis=0) - values.min(axis=0)
    inputs = np.column_stack([np.ones(len(data)), (values - values.min(axis=0)) / np.where(spread > 0, spread, 1)])
    growth = (complete["March2025"] - complete["March2024"]).clip(lower=0)
    state_totals = data.groupby("state_name")["existing_installs_count"].sum()
    within_state = data["existing_installs_count"] / data["state_name"].map(state_totals).replace(0, np.nan)
    weights = (data["state_name"].map(growth) * within_state).fillna(0).to_numpy()
    if weights.sum() <= 0:
        raise ValueError("The EIA baseline must have positive placement mass.")
    weights /= weights.sum()
    objectives = [Objective("Carbon", linear_objective, column=FIELDS[1]), Objective("Energy", linear_objective, column=FIELDS[2]), Objective("Racial equity", equity_objective, field="black_prop"), Objective("Income equity", equity_objective, field="Median_income")]
    return {"data": data, "inputs": inputs, "capacity": np.floor(data["count_qualified"]).to_numpy(dtype=np.int64), "weights": weights, "groups": data.groupby("state_name").indices, "objectives": objectives}


def evaluate(experiment, allocation):
    return np.array([objective.calc(experiment["data"], allocation) for objective in experiment["objectives"]])


def rank_allocation(scores, capacities, n):
    if n <= 0 or n > capacities.sum():
        raise ValueError("The installation budget must be positive and feasible.")
    order = np.argsort(-scores, kind="stable")
    remaining = n - np.concatenate(([0], np.cumsum(capacities[order][:-1])))
    allocation = np.zeros(len(capacities), dtype=np.int64)
    allocation[order] = np.minimum(capacities[order], np.maximum(remaining, 0))
    return allocation


def proportional_with_caps(weights, capacities, n):
    allocation = np.zeros(len(weights))
    active = (weights > 0) & (capacities > 0)
    if n > capacities[active].sum() + 1e-7:
        raise ValueError("The state-level baseline budget exceeds eligible capacity.")
    remaining = float(n)
    while remaining > 1e-7:
        shares = remaining * weights[active] / weights[active].sum()
        allocation[active] += np.minimum(shares, capacities[active] - allocation[active])
        remaining = n - allocation.sum()
        active = (weights > 0) & (capacities - allocation > 1e-8)
    return allocation


def baseline_allocation(experiment, n):
    allocation = np.zeros(len(experiment["data"]))
    for indices in experiment["groups"].values():
        weights = experiment["weights"][indices]
        amount = n * weights.sum()
        if amount > 0:
            allocation[indices] = proportional_with_caps(weights, experiment["capacity"][indices], amount)
    return allocation


def network_scores(network, inputs):
    values = {node: inputs[:, i] for i, node in enumerate(network.input_nodes)}
    values.update({node: np.zeros(len(inputs)) for node in network.output_nodes})
    for node, activation, aggregation, bias, response, links in network.node_evals:
        if aggregation.__name__ != "sum_aggregation":
            raise ValueError("This evaluator requires the configured sum aggregation.")
        total = np.zeros(len(inputs))
        for source, weight in links:
            total += weight * values[source]
        z = bias + response * total
        if activation.__name__ == "sigmoid_activation":
            output = 1 / (1 + np.exp(-np.clip(5 * z, -60, 60)))
        elif activation.__name__ == "relu_activation":
            output = np.maximum(0, z)
        else:
            output = np.fromiter((activation(float(value)) for value in z), dtype=float, count=len(z))
        values[node] = output
    return np.asarray(values[network.output_nodes[0]])


def train_eva(experiment, config_path, seed, selectivity, generations=6, n=2_000_000):
    random.seed(seed)
    np.random.seed(seed)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
    if len(config.genome_config.input_keys) != experiment["inputs"].shape[1]:
        raise ValueError("The source configuration and input vector disagree.")
    baseline = evaluate(experiment, baseline_allocation(experiment, n))
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
            allocation = rank_allocation(scores, experiment["capacity"], n)
            ratios.append(evaluate(experiment, allocation) / baseline)
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


def train_with_tuning(experiment, config_path, seed=0, max_rounds=6):
    selectivity = np.ones(4)
    baseline = evaluate(experiment, baseline_allocation(experiment, 2_000_000))
    log = []
    best_network = None
    best_cpr = -np.inf
    best_history = None
    columns = ["Stage", *["Selectivity: " + name for name in OBJECTIVES], *OBJECTIVES]
    for stage in range(max_rounds):
        network, history = train_eva(experiment, config_path, seed + 1000 * stage, selectivity)
        allocation = rank_allocation(network_scores(network, experiment["inputs"]), experiment["capacity"], 2_000_000)
        ratios = evaluate(experiment, allocation) / baseline
        log.append([stage, *selectivity, *ratios])
        if ratios.sum() > best_cpr:
            best_cpr = float(ratios.sum())
            best_network = network
            best_history = history
        if (ratios >= 1).all():
            return network, pd.DataFrame(log, columns=columns), history
        selectivity = np.where(ratios < 1, selectivity / np.maximum(ratios, 1e-9), selectivity)
    return best_network, pd.DataFrame(log, columns=columns), best_history


def eva_projections(experiment, network):
    scores = network_scores(network, experiment["inputs"])
    model_curves = {name: {} for name in OBJECTIVES}
    baseline_curves = {name: {} for name in OBJECTIVES}
    for n in range(100_000, 2_000_001, 100_000):
        allocation = rank_allocation(scores, experiment["capacity"], n)
        baseline = baseline_allocation(experiment, n)
        for name, value, reference in zip(OBJECTIVES, evaluate(experiment, allocation), evaluate(experiment, baseline)):
            model_curves[name][n] = float(value)
            baseline_curves[name][n] = float(reference)
    zip_codes = experiment["data"]["region_name"].to_numpy()
    model = Projection(model_curves, {int(z): int(a) for z, a in zip(zip_codes, allocation) if a > 0}, "Fresh EVA reconstruction")
    baseline = Projection(baseline_curves, {int(z): float(a) for z, a in zip(zip_codes, baseline) if a > 0}, "Reconstructed EIA status quo")
    ratios = pd.DataFrame(model.objective_projections) / pd.DataFrame(baseline.objective_projections)
    ratios.index.name = "Installations"
    return model, baseline, ratios, scores
