import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Utils"))
from paths import DATA, NEAT_CONFIG, SUNSIGHT
from q4_eva import OBJECTIVES, baseline_allocation, eva_projections, evaluate, network_scores, prepare_experiment, rank_allocation, train_with_tuning
from Simulation.projections_util import create_paper_objectives

raw = pd.read_csv(DATA / "data_by_zip.csv")
eia = pd.read_csv(DATA / "Q4_EIA_March_2024_2025.csv")
experiment = prepare_experiment(raw, eia)
networks = {}
endpoint_rows = []

for seed in [0, 1, 2]:
    network, tuning_log, generation_log = train_with_tuning(experiment, NEAT_CONFIG, seed=seed)
    networks[seed] = network
    model_projection, baseline_projection, ratios, scores = eva_projections(experiment, network)
    endpoint_rows.append([seed, *ratios.loc[2_000_000], int(ratios.ge(1).all(axis=1).sum())])

endpoint_summary = pd.DataFrame(endpoint_rows, columns=["Seed", *OBJECTIVES, "Budgets dominating all objectives"])
print(endpoint_summary.round(4).to_string(index=False))
model_projection, baseline_projection, ratios, scores = eva_projections(experiment, networks[0])

fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
for name in OBJECTIVES:
    ax.plot(ratios.index / 1_000_000, ratios[name], label=name, linewidth=2)
ax.axhline(1, color="black", linestyle="--")
ax.set(xlabel="Additional installation units (millions)", ylabel="Ratio to reconstructed status quo", title="Figure 3 reconstruction | fresh EVA, seed 0")
ax.legend()
ax.grid(alpha=0.2)
plt.show()

countries = pd.read_json(DATA / "countries.json", typ="series")
borders = pd.read_json(DATA / "state_lines.json", typ="series")

def draw_us_base(ax, countries, borders):
    for feature in countries['features']:
        if feature['properties'].get('ADM0_A3') == 'USA':
            geometry = feature['geometry']
            polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
            for polygon in polygons:
                ring = np.asarray(polygon[0])
                ax.fill(ring[:, 0], ring[:, 1], facecolor='#f2f4f6', edgecolor='#78828b', linewidth=0.7, zorder=0)
    for feature in borders['features']:
        if feature['properties'].get('ADM0_A3') == 'USA':
            geometry = feature['geometry']
            lines = geometry['coordinates'] if geometry['type'] == 'MultiLineString' else [geometry['coordinates']]
            for line in lines:
                xy = np.asarray(line)
                ax.plot(xy[:, 0], xy[:, 1], color='#b7bdc4', linewidth=0.5, zorder=1)
    ax.set_aspect(1 / np.cos(np.deg2rad(39)))
    ax.set(xlabel='Longitude', ylabel='Latitude')


data = experiment["data"]
valid = data["Latitude"].between(24, 50) & data["Longitude"].between(-125, -66)
percentiles = pd.Series(scores).rank(pct=True).to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), layout="constrained")
for ax, values, label in zip(axes, [scores, percentiles], ["Raw network score", "Score percentile (ranking diagnostic)"]):
    draw_us_base(ax, countries, borders)
    points = ax.scatter(data.loc[valid, "Longitude"], data.loc[valid, "Latitude"], c=values[valid], s=5, cmap="viridis", alpha=0.8, zorder=3)
    fig.colorbar(points, ax=ax, label=label, shrink=0.8)
    ax.set(xlim=(-125, -66), ylim=(24, 50), title=label)
fig.suptitle("Figure 4 reconstruction | ZIP scores, not allocations | seed 0", fontsize=13)
plt.show()

for n in ratios.index:
    allocation = rank_allocation(scores, experiment["capacity"], int(n))
    baseline = baseline_allocation(experiment, int(n))
    assert allocation.sum() == n
    assert np.isclose(baseline.sum(), n)
    assert (allocation >= 0).all() and (allocation <= experiment["capacity"]).all()
    assert (baseline >= 0).all() and (baseline <= experiment["capacity"] + 1e-6).all()
    equity = evaluate(experiment, allocation)[2:]
    assert ((equity >= 0) & (equity <= 1)).all()

native_scores = np.array([networks[0].activate(row)[0] for row in experiment["inputs"]])
assert np.allclose(native_scores, scores, atol=1e-14, rtol=0)
assert np.array_equal(rank_allocation(native_scores, experiment["capacity"], 2_000_000), rank_allocation(scores, experiment["capacity"], 2_000_000))
print("Budget, capacity, equity bounds, and native-inference checks passed.")

released_path = SUNSIGHT / "Models/Neat/models/05-31-25/NEAT_model_lexicase_weighted_5.pkl"
with released_path.open("rb") as stream:
    released_network = pickle.load(stream)

released_scores = network_scores(released_network, experiment["inputs"])
released_allocation = rank_allocation(released_scores, experiment["capacity"], 2_000_000)
reference = baseline_allocation(experiment, 2_000_000)
print("Released checkpoint ratios using the paper-equation reconstruction:")
print(pd.Series(evaluate(experiment, released_allocation) / evaluate(experiment, reference), index=OBJECTIVES).round(4))

zip_codes = experiment["data"]["region_name"].to_numpy()
placed = {int(z): float(a) for z, a in zip(zip_codes, released_allocation) if a > 0}
baseline_placed = {int(z): float(a) for z, a in zip(zip_codes, reference) if a > 0}
print("Native toolkit metric ratios for the same allocations:")
for objective in create_paper_objectives():
    print(objective.name, objective.calc(experiment["data"], placed) / objective.calc(experiment["data"], baseline_placed))
