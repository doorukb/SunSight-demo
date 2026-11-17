import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Utils"))
from paths import DATA, NEAT_CONFIG, Q4, Q5
from q4_eva import OBJECTIVES, baseline_allocation, evaluate, network_scores, prepare_experiment, rank_allocation
from q5_softmax import evaluate_policy, softmax_allocation, softmax_path, train_softmax_with_tuning

models, results, plots = Q5 / "Models", Q5 / "Results", Q5 / "Plots"
experiment = prepare_experiment(pd.read_csv(DATA / "data_by_zip.csv"), pd.read_csv(DATA / "Q4_EIA_March_2024_2025.csv"))
temperature = 0.01
evolution_seeds = [0, 1, 2]
training_seeds = (10, 11, 12, 13)
test_seeds = range(20_000, 20_100)
q4_models = {seed: pickle.loads((Q4 / "Models" / f"eva_seed_{seed}.pkl").read_bytes()) for seed in evolution_seeds}


softmax_models = {}
training_rows = []
for seed in evolution_seeds:
    network, tuning, history = train_softmax_with_tuning(experiment, NEAT_CONFIG, seed, temperature, training_seeds, max_rounds=6)
    softmax_models[seed] = network
    (models / f"softmax_seed_{seed}.pkl").write_bytes(pickle.dumps(network))
    tuning.to_csv(results / f"tuning_seed_{seed}.csv", index=False)
    history.to_csv(results / f"training_seed_{seed}.csv", index=False)
    passed = tuning[OBJECTIVES].ge(1).all(axis=1)
    selected = int(tuning.index[passed][0]) if passed.any() else int(tuning[OBJECTIVES].sum(axis=1).idxmax())
    training_rows.append({"Evolution seed": seed, "Tuning rounds": len(tuning), "Selected stage": selected, "Training criterion met": bool(passed.any())})
training_status = pd.DataFrame(training_rows)
training_status.to_csv(results / "training_status.csv", index=False)
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
endpoint_samples.to_csv(results / "endpoint_samples.csv", index=False)
means.to_csv(results / "endpoint_means.csv")
sds.to_csv(results / "endpoint_deployment_sd.csv")
print(means.round(6).to_string())
print(sds[OBJECTIVES].round(6).to_string())


rows = []
for t in [1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]:
    samples = evaluate_policy(experiment, q4_models[0], t, test_seeds)
    samples["Temperature"] = t
    rows.append(samples)
temperature_samples = pd.concat(rows, ignore_index=True)
temperature_samples.to_csv(results / "temperature_samples.csv", index=False)
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
fig.savefig(plots / "temperature_sensitivity.png", dpi=180)
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
path_samples.to_csv(results / "nested_path_samples.csv", index=False)
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
fig.savefig(plots / "nested_deployment.png", dpi=180)
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

