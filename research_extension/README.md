# Research extension

Replications and extensions of the SunSight papers, one folder per question.

```
research_extension/
├── Data/                  shared inputs: data_by_zip.csv, clean_data_by_zip.csv (written by Q2),
│                          Q4_EIA_March_2024_2025.csv, countries.json, state_lines.json, states.json
├── Utils/                 shared code: paths.py (locations + sys.path), q4_eva.py, q5_softmax.py, q6_state_eva.py
├── SunSight_Q3/           vendored SunSight toolkit, coopersigrist/SunSight @ 99588d1696c980075a19abf087c669ae5aa22e4e
├── Q2_SunSight_research_extension/   data analysis + extensions 1-6 notebook
├── Q4_EVA_reconstruction/            EVA reproduction and audit
├── Q5_softmax_EVA/                   stochastic softmax placement
└── Q6_state_EVA/                     200 state objectives with sampled lexicase selection
```

Q4-Q6 share one layout: `README.md`, `requirements.txt`, the notebook, a `run_Q*.py` script with the same steps,
and generated output in `Models/` (pickled networks), `Results/` (CSV/JSON), `Plots/` and `Logs/`.

## Running

Install each question's `requirements.txt`, then open the notebook from its own folder (paths are resolved
through `Utils/paths.py`) or run `python run_Q4.py` / `run_Q5.py` / `run_Q6.py`. Nothing is downloaded: the pinned
SunSight toolkit and all data are in the repository. Q5 reads Q4's networks from `Q4_EVA_reconstruction/Models/`.

Pickled networks are locally generated artifacts; load only trusted ones. The Q2 notebook retrains Q5's
networks, so it rewrites `Q5_softmax_EVA/Results/tuning_seed_*.csv` and `training_seed_*.csv` (deterministic, so
the content is unchanged).
