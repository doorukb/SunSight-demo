## Extension 4 — EVA reproduction and reproducibility audit

This notebook reconstructs the paper's described EVA workflow using pinned SunSight data/configuration, its objective and projection interfaces, and NEAT-Python genome operations. It regenerates the two requested figure types. It does **not** numerically reproduce the published curves or establish that a released checkpoint is the exact figure-generating model.

Figure 3 compares Carbon Offset, Energy Generation, Racial Equity, and Income Equity with a reconstructed EIA status quo at each positive budget. Figure 4 maps the scoring network's outputs, not allocated installation counts.

### Objective definitions

The count-balance equity objectives after additions are, with $b_i=I_i+a_i$ representing existing-plus-new installations,

$$f_{IE}(a)=1-\frac{|\sum_{i\in H_I}b_i-\sum_{i\in L_I}b_i|}{\sum_i b_i},$$

$$f_{RE}(a)=1-\frac{|\sum_{i\in H_R}b_i-\sum_{i\in L_R}b_i|}{\sum_i b_i}.$$

This follows Section 6.2's description of evaluating the mix after new installations are factored in; an added-only equity calculation would be a different experiment. Median ties go into the high group. The linear objectives use the released toolkit indicators `carbon_offset_kg_per_panel` and `yearly_sunlight_kwh_kw_threshold_avg`. Ratios are reported; these legacy indicators and allocation caps are not reinterpreted as a physically verified individual-panel inventory.

$$CPR(a)=\sum_{k=1}^4\frac{f_k(a)}{f_k(a^{SQ})}.$$

CPR above four does not imply that every objective exceeds its baseline.

### Training

There are five measured attributes, min-max normalized: income, carbon, solar resource, stored panel utilization, and Black population proportion. A constant leading input reproduces the released six-input configuration's bias expansion. It is not a sixth measured attribute.

Population: 30; generations: six; training budget: two million; survival target: 0.3; one selected elite. For objective selectivities $v_k$, each sequential filter retains approximately

$$s_k=0.3^{v_k/\sum_jv_j}$$

of the current survivors. Objectives are randomly ordered. Survivors are finally ranked by CPR. Twenty-nine distinct selected-parent pairs generate mutated offspring, plus one elite. The best selected CPR checkpoint seen during the run is retained. The global parent-pool reconstruction does not reproduce NEAT-Population speciation or call the released binary-fitness training wrapper unchanged.

Adaptive selectivity starts at one for each objective and increases the weight of objectives below baseline. At most six tuning rounds are permitted; all three reported seeds passed at the first round. Seed 0 was designated as the primary figure before comparing seeds; seeds 1 and 2 are diagnostics.

### Baseline reconstruction

The supplementary CSV is transcribed from EIA's May 2025 Electric Power Monthly, Table 6.2.B: March 2025 and March 2024 estimated small-scale PV capacity, all sectors. The paper specifies March-to-March, whereas the released monthly CSV stops at February 2025 and its helper uses January residential capacity. The precise historical extraction underlying the figures is not identified.

Negative net changes (Rhode Island and Missouri) are clipped to zero. Alabama is marked NM (not meaningful due to large relative standard error) and is excluded rather than imputed. State growth shares are combined with within-state existing-installation shares and redistributed within states where caps bind. This yields 10,243 eligible ZIP codes. These are declared reconstruction choices, not an assertion that the authors made identical choices.

The legacy SunSight `count_qualified` cap is retained for this separate reproduction experiment. It differs from Question 2's estimated individual-panel capacity. There is no claim that the numerical energy or capacity units match Questions 2–3.

### Metric drift and reproduction status

The released `calc_equity` implementation uses existing-plus-new installation density per potential panel and a formula of the form $2-|\bar u_H-\bar u_L|/\bar u$, not the count-balance equations in paper Section 4.2. Those metrics must not silently be interchanged.

The selected released checkpoint is `Models/Neat/models/05-31-25/NEAT_model_lexicase_weighted_5.pkl`. Its name does not establish that it generated the paper's figures. Replaying it in this reconstructed experiment does not reproduce four-objective dominance.

For fresh training, seed 0 has endpoint ratios approximately (1.2880, 1.1944, 1.0218, 1.3901). It exceeds status quo on all objectives at only seven of twenty evaluated budgets. The paper's all-budget dominance is therefore not reproduced here.

The reconstructed racial-equity baseline at two million is approximately 0.8802. Since the count-balance objective is bounded by one, the maximum possible ratio under these conditions is $1/0.8802\approx1.1361$. The paper's approximately 1.24 reported racial-equity ratio cannot be matched under this exact reconstructed denominator. This shows that the evaluation conditions differ; it does not by itself establish an error in the paper.

Raw sigmoid scores concentrate near one. They are ranking values, not probabilities. The figure shows raw scores plus a clearly labeled percentile diagnostic; percentile coloring does not change allocations.

### Model inspection

The primary seed-0 network contains one sigmoid output and no hidden nodes. Its ranking is therefore equivalent to a linear combination of the retained normalized inputs. The experiment does not demonstrate an advantage from learned nonlinear ranking structure. The equity metrics compare ZIP groups, not the reported race or income of individual solar owners.

### References

- Paper 2, Sections 4.2–5.2 and 7.1: https://doi.org/10.1145/3736425.3770115
- Pinned toolkit: https://github.com/coopersigrist/SunSight/tree/99588d1696c980075a19abf087c669ae5aa22e4e
- Archived EIA input, Table 6.2.B: https://www.eia.gov/electricity/monthly/archive/may2025.pdf
- NEAT-Python: https://neat-python.readthedocs.io/
