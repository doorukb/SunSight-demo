# Research extension Q6: 200 state objectives and sampled lexicase selection

The requested change creates **four objective families times 50 states = 200 objectives**, not 50 objectives total. We retain one shared ZIP-scoring network and one national allocation budget. The network still returns one score per ZIP; the evaluator now returns a 50-by-4 fitness matrix. No state quotas or 50 separate networks are introduced.

This continues our Q4/Q5 methodological reconstruction using NEAT genome crossover and mutation operators, not a verified numerical replication of EVA. The reconstructed evolutionary loop has no NEAT species allocation. Paper 2 motivates the scoring, objective and evolutionary framework; the state decomposition and experiments here are our extension. We retain Q5's capacity-aware raw-score softmax at temperature $T=0.01$.

## 1. All 50 states and an explicit reference

The supplied CSV contains 10,476 eligible ZIPs in the 50 states. Exclude the District of Columbia and the Puerto Rico municipal labels. The original CSV is not exhaustive coverage of every ZIP in each state.

Q4's EIA-based preparation omitted Alabama because its baseline value was missing. Reusing that preparation would silently create only 196 objectives. We instead include Alabama and define a capacity-proportional reference allocation:

$$
b_i=N\frac{u_i}{\sum_j u_j},\qquad N=2{,}000{,}000,\qquad u_i=\lfloor\mathrm{count\_qualified}_i\rfloor.
$$

This is a fractional benchmark, not a sampled deployment or a forecast of adoption. It has total $N$ and satisfies $0\le b_i\le u_i$. Every state has positive reference carbon, energy and equity in these data. We do not impute an EIA value or add arbitrary smoothing constants to denominators. The state list is explicit in `states.json`; no EIA numerical values are used in this experiment.

All methods in this question use the same all-50-state data, input normalization, placement policy, reference and computational budget. Their ratios must not be compared numerically with Q5's differently defined baseline. Toolkit installation-unit and energy/carbon indicator conventions remain as in Q4, so these results are not engineering forecasts.

## 2. Define the state objectives

Let $Z_s$ be the ZIPs in state $s$, $a_i$ the additional allocation and $I_i$ the existing installations. With per-unit toolkit indicators $c_i$ and $e_i$, define

$$
C_s(a)=\sum_{i\in Z_s}a_i c_i,\qquad E_s(a)=\sum_{i\in Z_s}a_i e_i.
$$

For income and Black-population-proportion equity, partition ZIPs using the respective **within-state median**. ZIPs at the median enter the high group. Define $H_{s,k}$ and $L_{s,k}$ for demographic family $k$, and

$$
Q_{s,k}(a)=1-\frac{\left|\sum_{i\in H_{s,k}}(I_i+a_i)-\sum_{i\in L_{s,k}}(I_i+a_i)\right|}{\sum_{i\in Z_s}(I_i+a_i)}.
$$

Within-state medians make these measures local distributional comparisons. They are a stated modeling choice: a ZIP can be high relative to its state and low relative to the country. Consequently, these are not a partition of the original national demographic cohorts. Equal counts across two ZIP groups are a demographic proxy, not a guarantee of household-level fairness.

Both groups exist and total existing installations are positive in every state in these data. A state receiving no new units has zero new carbon and energy, while its equity remains its existing-installation equity. It is not assigned perfect equity or a missing value.

Normalize each objective against the same state's reference:

$$
R_{s,k}(a)=\frac{F_{s,k}(a)}{F_{s,k}(b)}.
$$

The 200-vector uses state-major order, with families carbon, energy, racial equity and income equity. For $s=0,\ldots,49$ and $k=0,\ldots,3$, case index $j=4s+k$.

## 3. Sample 40 cases per generation

At each generation, uniformly sample 10 distinct states without replacement. Include all four families for each selected state. This produces 40 cases and prevents a generation's subset from containing only energy or only equity cases. It also makes the state inclusion probability exactly $10/50=0.2$. The subset is shared by all candidates and all nine survivor-selection events in that generation. Each event shuffles its case order independently.

The subset changes between generations. Independent sampling does not guarantee coverage: a particular state is never sampled over $G$ generations with probability $0.8^G$. After 30 generations the expected number of unsampled states is $50(0.8)^{30}\approx0.0619$. Actual coverage must be logged rather than presumed.

## 4. Semi-dynamic epsilon-lexicase survival

Use continuous-valued epsilon-lexicase instead of exact-best filtering. For every case $j$, calculate one tolerance from the full generation's population:

$$
\epsilon_j=\operatorname{median}_g\left|R_{g,j}-\operatorname{median}_h R_{h,j}\right|.
$$

For each survivor slot, start with all candidates not already selected and shuffle the sampled cases. For case $j$ and current pool $P$, retain

$$
P\leftarrow\left\{g\in P:R_{g,j}\ge\max_{h\in P}R_{h,j}-\epsilon_j\right\}.
$$

Stop when one remains or the cases are exhausted, then choose uniformly among remaining candidates. Repeat to obtain nine distinct survivors. Removing already selected winners is our without-replacement survivor variant; it preserves Q4's pool of distinct parents for crossover. Standard lexicase parent sampling is commonly with replacement, so its selection guarantees should not be transferred unchanged to this variant.

A case tied across the population leaves the pool unchanged. If MAD is zero but candidates differ, filtering uses the exact best score. No arbitrary ordering is used to discard identical candidates.

Positive affine rescaling of an individual case rescales its MAD by the same factor and leaves its filtering decisions unchanged, apart from numerical precision. Thus carbon's numerical units do not automatically dominate the selector. Normalization matters for scalar model ranking, crossover fitness and interpretation, rather than being necessary to make case-by-case filtering unit invariant.

## 5. Keep scalar preferences visible

Lexicase survival uses the case vector, but the reconstructed EVA loop still needs scalar decisions for checkpoint choice, the retained elite and the fitter parent in NEAT crossover. For the state methods, begin with

$$
S_{\mathrm{CPR}}=\frac{4}{200}\sum_{s,k}R_{s,k}.
$$

This equals the mean of 50 state CPRs and equals four at the reference. It weights states equally, not by population, capacity or national energy contribution. We choose the highest-scoring member of the nine selected survivors as the generation's elite, and return the best selected checkpoint across generations. The national control instead uses the sum of its four national ratios.

Initial diagnostics showed that a few extremely large state ratios can dominate this scalar score even when most states receive little benefit. We therefore test a separate exploratory variant:

$$
S_{\log}=\frac{4}{200\log 2}\sum_{s,k}\log(1+R_{s,k}).
$$

Its marginal reward is proportional to $1/(1+R)$, so increasing an already large ratio receives less reward. It is finite at zero and still equals four at the reference. The log is applied to the estimated mean ratios; lexicase still filters on the original case ratios and MAD tolerances. This change affects checkpoint ranking, elitism and crossover fitness, not only reporting. It is not a proof that all states improve, nor a max-min fairness constraint.

## 6. Experimental controls and evaluation

Compare five methods:

| Method | Survival cases | Scalar checkpoint/crossover score |
|---|---|---|
| National 4 | Four national objectives, using national medians | National CPR |
| State means 4 | Four equal-state averages of state ratios | State CPR |
| State 200 | All 200 state-family cases | State CPR |
| State subset 40 | Four cases for each of 10 sampled states | State CPR |
| Subset 40 + log score | The same sampled-case design | Logarithmic state score |

The state-means control separates averaging away state-specific performance from the state decomposition itself. Comparing State 200 and State subset 40 isolates case subsampling. Comparing the two subset methods isolates the scalar-score change. National versus state methods also differ in demographic thresholds and scalar geographic preferences, so that comparison is not a one-factor causal test.

All runs use 30 networks, 30 generations, nine distinct survivors, one selected elite and 29 offspring from distinct shuffled survivor pairs. This is a fixed-budget experiment without Q5's outer selectivity-tuning loop. Each candidate is evaluated on placement seeds 10, 11, 12 and 13, averaging realized objective values before normalization. State equity has a random state-specific denominator, so we do not reuse Q5's fixed-denominator Jensen argument.

Use evolutionary seeds 0 through 4, with seed 0 as the primary illustration. Initial diagnostics used placement seeds 20000 through 20099 and motivated the log-score variant. Final tables evaluate every method again using fresh placement seeds **40000 through 40099**, which were not used for training, checkpoint selection or the earlier diagnostics. These are held-out deployment draws on the same dataset, not held-out states or evidence of geographic generalization. The additional variant is exploratory rather than a preregistered confirmatory result.

First average 100 deployment draws for each learned network, then summarize those five network means. Between-training-seed standard deviations are stored separately from deployment variation. State success counts refer to all four held-out mean ratios reaching one, not to a statistical significance test or a guarantee for every deployment.

We evaluate all 200 scores for checkpointing and diagnostics, even when survival uses only 40. Thus this implementation does not claim the simulation-cost savings sometimes associated with down-sampled lexicase. The expensive global softmax allocation is still performed over all eligible ZIPs.
