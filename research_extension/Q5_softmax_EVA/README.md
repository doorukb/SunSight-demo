Install dependencies with `pip install -r requirements.txt`, and open Q5_softmax_EVA.ipynb from this folder or run `python run_Q5.py`. The pinned SunSight toolkit (`../SunSight_Q3`), shared inputs including the baseline transcription (`../Data`) and shared helpers `q4_eva.py` / `q5_softmax.py` (`../Utils`) ship with the repository, so no git or internet access is needed. Saved pickle files (`Models/`, plus the Q4 controls in `../Q4_EVA_reconstruction/Models/`) are locally generated network artifacts; load only trusted artifacts. All Q5 training is rerun. Outputs go to `Results/`, `Plots/` and `Logs/`. This extends the Q4 reconstruction, not a verified replication of the original paper.

## Results and interpretation

At two million additional installation units, held-out means for primary evolutionary seed 0 are:

| Policy | Carbon | Energy | Racial equity | Income equity | ZIPs used |
|---|---:|---:|---:|---:|---:|
| Q4 greedy | 1.2880 | 1.1944 | 1.0218 | 1.3901 | 289 |
| Q4 network + softmax | 1.1250 | 0.9731 | 1.1155 | 1.4272 | 10,210.1 |
| Retrained softmax EVA | 1.5533 | 0.9422 | 1.1257 | 1.5282 | 3,508.1 |

All four objective columns are ratios to the reconstructed status quo, so one is the baseline and larger is better. Fractional ZIP counts are averages across draws.

Changing only placement disperses the Q4 seed-0 network across almost every ZIP, lowering both physical-indicator ratios while improving the two equity ratios. Retraining changes score gaps and produces a more concentrated stochastic policy. It increases carbon and both equity ratios relative to greedy Q4 for this seed, but energy is 5.78% below baseline. It does not dominate Q4 or the baseline across all objectives.

| Evolutionary seed | Retrained carbon | Energy | Racial equity | Income equity | Tuning stages | Training criterion met |
|---|---:|---:|---:|---:|---:|---|
| 0 | 1.5533 | 0.9422 | 1.1257 | 1.5282 | 6 | No |
| 1 | 1.5951 | 1.0199 | 1.0852 | 1.4940 | 1 | Yes |
| 2 | 1.4450 | 1.0006 | 1.0567 | 1.4926 | 2 | Yes |

Seed 0 selects stage 2 (the best estimated CPR), not the final failed stage. Seeds 1 and 2 achieve all four baseline ratios in held-out means at the final budget; that does not establish all-budget or every-realization dominance. For retrained seed 0, deployment SDs are approximately 0.000063, 0.000067, 0.000550 and 0.000730 in the four columns. Variation across learned networks is much larger here. Three evolutionary seeds are insufficient to establish general superiority.

The fixed Q4 seed-0 temperature sweep reinforces the calibration issue. At $T=10^{-10}$ it matches greedy allocation in this dataset. At $T=10^{-4}$ it uses about 9,058 ZIPs, with ratios approximately $(1.1951,1.0009,1.0461,1.5447)$. At $T=1$ it uses all 10,243 ZIPs and yields approximately $(1.1000,0.9658,1.0971,1.4089)$. Equity is not monotonic in temperature. Geographic dispersion alone is not a demographic fairness guarantee.

Softmax is a valid controllable stochastic alternative, and retraining is more coherent than assuming ordinal greedy scores are already calibrated probabilities. These experiments show tradeoffs and sensitivity, rather than evidence that softmax is inherently better. The fixed temperature does not guarantee fixed entropy across networks, and the small reused training seed pool may be overfit. A later experiment could compare score-scale controls or entropy targets using a separate validation set; those alternatives are not silently substituted for the raw-score policy requested here.

## Sources and reproducibility

The framework is grounded in the supplied *Multi-objective Evolutionary Learning for Near Pareto-Optimal Optimization of Solar Deployment* (DOI: 10.1145/3736425.3770115), with the siting/equity context supplied by *Lost in Siting: The Hidden Carbon Cost of Inequitable Residential Solar Installations*. The present stochastic rule, implementation and experiments are our extension, not reported results from those papers. Q4 reconstruction limitations continue to apply.

- Pinned source: https://github.com/coopersigrist/SunSight/tree/99588d1696c980075a19abf087c669ae5aa22e4e
- NumPy multinomial definition: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multinomial.html
- Stable softmax documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html

The sampler was compared with exact enumeration on a small capacity-limited example using 50,000 trials (maximum absolute state-frequency discrepancy 0.00273). Additional checks cover exact budgets, integer counts, capacities, repeatability, native-network inference agreement and nested nonrelocating deployment paths. The deterministic training branch reproduces Q4 seed 0's network scores exactly. These checks validate implementation behavior, not the empirical truth of the data or numerical replication of the original paper.
