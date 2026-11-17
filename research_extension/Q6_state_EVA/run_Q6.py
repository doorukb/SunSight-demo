import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Utils"))
from paths import DATA, NEAT_CONFIG, Q6
from q4_eva import OBJECTIVES, network_scores
from q5_softmax import softmax_allocation
from q6_state_eva import evaluate_state_model, national_values, prepare_states, sample_cases, state_values, train_state_eva
from report_Q6 import plot_results, summarize_results

models_dir, results, runs, plots = Q6 / "Models", Q6 / "Results", Q6 / "Results" / "runs", Q6 / "Plots"

state_names = json.loads((DATA / "states.json").read_text())
experiment = prepare_states(pd.read_csv(DATA / "data_by_zip.csv"), state_names)
print("ZIPs:", len(experiment["data"]), "States:", len(experiment["states"]), "Objectives:", experiment["state_reference"].size)


retrain = False
models = {}
for label in ["national", "state_mean", "state_all", "state_subset", "state_subset_log"]:
    for seed in range(5):
        path = models_dir / f"{label}_seed_{seed}.pkl"
        if retrain:
            mode = "state_subset" if label == "state_subset_log" else label
            checkpoint = "log" if label == "state_subset_log" else "cpr"
            network, history, subsets, events = train_state_eva(experiment, NEAT_CONFIG, seed, mode, checkpoint=checkpoint)
            path.write_bytes(pickle.dumps(network))
            history.to_csv(runs / f"{label}_history_{seed}.csv", index=False)
            subsets.to_csv(runs / f"{label}_subsets_{seed}.csv", index=False)
            events.to_csv(runs / f"{label}_events_{seed}.csv", index=False)
        else:
            network = pickle.loads(path.read_bytes())
        models[(label, seed)] = network


for (label, seed), network in models.items():
    national, local = evaluate_state_model(experiment, network, deployment_seeds=range(40_000, 40_100))
    national.to_csv(runs / f"{label}_national_test_{seed}.csv", index=False)
    local.to_csv(runs / f"{label}_state_test_{seed}.csv", index=False)


endpoint, state, coverage = summarize_results(runs, results)
columns = ["National " + name for name in OBJECTIVES] + ["State mean " + name for name in OBJECTIVES] + ["States meeting all four in held-out means"]
print(endpoint.groupby("Mode")[columns].mean().round(4).to_string())
print(endpoint.groupby("Mode")[columns].std().round(4).to_string())
print(coverage.to_string(index=False))
plot_results(plots, endpoint, state)


assert len(experiment["states"]) == 50 and experiment["state_reference"].shape == (50, 4)
assert np.allclose(state_values(experiment, experiment["reference"]) / experiment["state_reference"], 1)
scores = network_scores(models[("state_subset_log", 0)], experiment["inputs"])
allocation = softmax_allocation(scores, experiment["capacity"], experiment["n"], 0.01, np.random.default_rng(12345))
assert allocation.sum() == experiment["n"] and (allocation <= experiment["capacity"]).all()
assert np.allclose(state_values(experiment, allocation)[:, :2].sum(axis=0), national_values(experiment, allocation)[:2])
cases = sample_cases(np.random.default_rng(5), 10)
assert len(cases) == 40 and len(set(cases // 4)) == 10
print("Objective dimensions, reference identity, exact budget, capacity, physical additivity and subset checks passed.")

