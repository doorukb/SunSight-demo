# Q6: State objectives

Extract the folder, use Python 3.11 or later, and install dependencies with `pip install -r requirements.txt`. Run Q6_state_objectives.ipynb or `python run_Q6.py` from the extracted folder. Git and internet are needed on first use to obtain the pinned SunSight source. The default reruns held-out evaluation using the included locally generated model artifacts. Set `retrain = True` in the notebook or script to retrain all five methods on all five evolutionary seeds. Load only trusted pickle artifacts.

The notebook expands all new Q6 functions. q4_eva.py and q5_softmax.py contain inherited helpers. methodology.md and results.md contain notebook-ready text and mathematics. Final *_test_*.csv files use seeds 40000–40099; older training logs print the exploratory 20000–20099 diagnostics. Those preliminary numbers are not the final results.

## Results

All values below use the fresh final deployment seed pool. Ratios use this question's capacity-proportional reference, not Q5's status-quo reference. Each row first averages 100 deployment draws per model, then averages five evolutionary seeds.

| Method | National carbon | National energy | National racial equity | National income equity | Mean states meeting all four |
|---|---:|---:|---:|---:|---:|
| National 4 | 1.4763 | 1.0199 | 0.9807 | 1.0986 | 0.0 |
| State means 4 | 0.5615 | 0.8807 | 0.4456 | 0.8196 | 5.8 |
| State 200 | 0.5493 | 0.8632 | 0.5671 | 0.8532 | 6.0 |
| State subset 40 | 0.6435 | 0.8783 | 0.6357 | 0.9128 | 6.6 |
| Subset 40 + log score | 1.0435 | 0.9520 | 0.9632 | 1.0596 | 22.8 |

| Method | Mean state carbon | Mean state energy | Mean state racial equity | Mean state income equity | Mean of per-run median state energy |
|---|---:|---:|---:|---:|---:|
| National 4 | 1.5372 | 1.4289 | 0.8514 | 0.8299 | 0.0000 |
| State means 4 | 2.5156 | 2.5710 | 0.9201 | 0.8501 | 0.0784 |
| State 200 | 2.1351 | 2.1742 | 1.0132 | 0.9280 | 0.1419 |
| State subset 40 | 1.9193 | 1.9466 | 1.0962 | 0.9805 | 0.3148 |
| Subset 40 + log score | 1.3244 | 1.3287 | 1.2456 | 1.0794 | 1.2588 |

### Interpret the distributions, not only their means

The original State subset 40 method has a mean state energy ratio of about 1.9466, yet its mean of per-run median state energy ratios is only 0.3148. In the primary seed-0 run, 42 states have mean energy below reference, despite mean state energy of 2.0237. Large gains in a few states dominate that average. Equal state weighting is not itself a lower-tail guarantee.

The log-score variant reduces that compensation effect. Its mean of per-run median state energy ratios is 1.2588; the average number of states meeting all four held-out mean objectives rises from 6.6 to 22.8. Every sampled deployment in these log-score runs allocates to all 50 states. In the primary run, 25 states meet all four means, compared with zero for the original subset run. Between-training-seed SD of the success count is about 4.49 for the log-score method, versus 5.08 for the original subset method.

This is a supported improvement in the measured state-coverage criteria within these experiments, not general superiority. The log-score method's national energy ratio averages 0.9520 and national racial equity 0.9632, both below reference. Its national carbon mean is 1.0435 and national income-equity mean 1.0596. National 4 has substantially higher national carbon and energy. No method establishes improvement on every state and national objective.

State 200 versus State subset 40 also shows a tradeoff, rather than a universal win for subsampling: the latter improves several equity/coverage summaries while lowering the mean state carbon and energy ratios. Five evolutionary seeds are too few to establish a general statistical ranking. Full-precision per-run means and between-seed SDs are supplied in the CSV files.

### Why national and average-state results differ

For energy, the national ratio is a reference-energy-weighted average of state ratios:

$$
\frac{\sum_s E_s(a)}{\sum_s E_s(b)}
=
\sum_s\left(\frac{E_s(b)}{\sum_t E_t(b)}\right)R_{s,E}(a).
$$

The mean state ratio instead weights every state by $1/50$. They measure different things. A large relative gain in a state with little reference energy can noticeably increase the state mean while contributing little to national energy. The same identity holds for carbon. National demographic equity is nonlinear and uses different cohorts, so it is not the arithmetic mean of within-state equity scores.

### Selection diagnostics and limitations

Across the five original subset runs, 50, 50, 50, 50 and 49 states were ever sampled. Minnesota was not directly sampled in evolutionary seed 4. It still contributed to the full-vector scalar checkpoint score; direct lexicase exposure and checkpoint influence are different. A future coverage-controlled design could shuffle all 50 states and distribute them across five generations before reshuffling.

The main subset runs examined about 5.76 to 8.70 cases per survivor slot on average, although 40 were available. Later cases are often unnecessary once the pool reaches one candidate, while many examined cases do not change the pool. Logs record both exposure and actual pool reduction. Being sampled is not the same as materially influencing selection.

The shared network has the same five measured features plus the legacy constant input; no state identifier was added. It cannot freely learn 50 unrelated policies when ZIPs have overlapping attributes. Small populations, the four-draw training seed pool and the restricted scoring-function class remain limitations. Held-out placement seeds assess stochastic deployment, not uncertainty in the input data or generalization to new geographies.

## Sources

- Supplied paper: *Multi-objective Evolutionary Learning for Near Pareto-Optimal Optimization of Solar Deployment*, Sections 4–5, DOI https://doi.org/10.1145/3736425.3770115.
- Supplied paper: *Lost in Siting: The Hidden Carbon Cost of Inequitable Residential Solar Installations*, demographic and geographic siting context.
- La Cava et al., *A probabilistic and multi-objective analysis of lexicase selection and epsilon-lexicase selection*: https://arxiv.org/abs/1709.05394. The continuous-score filtering and population-MAD tolerance are literature-based; this implementation and state experiment are constructed here.
- Helmuth and Spector, *Problem-solving benefits of down-sampled lexicase selection*: https://arxiv.org/abs/2106.06085. Published benefits in genetic programming are motivation, not evidence of a solar-deployment benefit.
- Pinned SunSight source: https://github.com/coopersigrist/SunSight/tree/99588d1696c980075a19abf087c669ae5aa22e4e.
