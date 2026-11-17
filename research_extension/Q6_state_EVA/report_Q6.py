from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q4_eva import OBJECTIVES


def summarize_results(runs, results):
    runs, results = Path(runs), Path(results)
    endpoint = []
    state = []
    coverage = []
    for mode in ["national", "state_mean", "state_all", "state_subset", "state_subset_log"]:
        for seed in range(5):
            national = pd.read_csv(runs / f"{mode}_national_test_{seed}.csv")
            local = pd.read_csv(runs / f"{mode}_state_test_{seed}.csv")
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
                subsets = pd.read_csv(runs / f"{mode}_subsets_{seed}.csv")
                events = pd.read_csv(runs / f"{mode}_events_{seed}.csv")
                counts = subsets.assign(StateIndex=subsets["Case"] // 4).groupby("StateIndex").size().reindex(range(50), fill_value=0) / 4
                effective = events.loc[events["After"] < events["Before"]].assign(StateIndex=lambda x: x["Case"] // 4).groupby("StateIndex").size().reindex(range(50), fill_value=0)
                depths = events.groupby(["Generation", "Slot"]).size()
                coverage.append({"Mode": mode, "Evolution seed": seed, "States ever sampled": int((counts > 0).sum()), "Minimum sampled generations": counts.min(), "Maximum sampled generations": counts.max(), "Mean cases examined per slot": depths.mean(), "Fraction examined cases reducing pool": (events["After"] < events["Before"]).mean(), "States ever reducing a pool": int((effective > 0).sum())})
    endpoint = pd.DataFrame(endpoint)
    state = pd.concat(state, ignore_index=True)
    coverage = pd.DataFrame(coverage)
    endpoint.to_csv(results / "per_run_summary.csv", index=False)
    state.to_csv(results / "state_heldout_means.csv", index=False)
    coverage.to_csv(results / "selection_diagnostics.csv", index=False)
    metrics = endpoint.select_dtypes(include=np.number).columns.drop("Evolution seed")
    endpoint.groupby("Mode")[metrics].mean().to_csv(results / "five_seed_means.csv")
    endpoint.groupby("Mode")[metrics].std().to_csv(results / "five_seed_sd.csv")
    return endpoint, state, coverage


def plot_results(plots, endpoint, state):
    plots = Path(plots)
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
    fig.savefig(plots / "national_vs_state.png", dpi=180)
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
    fig.savefig(plots / "state_distribution.png", dpi=180)
    plt.show()
    plt.close(fig)
