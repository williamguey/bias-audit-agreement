# From Bias Scores to Model Rankings

**A Measurement Audit of Ten LLM Probe Families**

Rebuilt manuscript, reproducible reanalysis, and original response data. This
revision audits the implementation behind the earlier preprint *Bias Audits
Detect Bias but Disagree on Ranking*. It changes that paper's interpretation;
it is not a cosmetic rewrite. The manuscript is an **author-review draft**, with
AI assistance disclosed. It has not been submitted or approved by all authors.

## What the revised evidence supports

The release contains 19,500 stored call records from ten adapted probe families,
ten original models, six additional models, and gender, age, and socioeconomic
cue sets. These short custom probes are not implementations of the complete
benchmarks whose names appear in the historical files.

- Keeping matched scenarios together changes split-half rank stability. Under
  identical historical scoring and response weights, decision contrasts change
  from mean Spearman r = -0.495 to 0.048. This does not establish dependable ranks.
- The primary revised absolute-net comparison has Kendall's W = 0.230
  (permutation p = 0.027; nine probes, nine complete-case models). Signed-net
  scores give W = 0.114 (p = 0.423). Removing mirrored assertions gives
  W = 0.189 (p = 0.139). These are exploratory specifications, not a set of
  threshold-based discoveries or a common fairness scale.
- A confidence interval above zero for an absolute estimated score does not by
  itself establish detection of an underlying nonzero effect. Conditional
  sign-symmetry diagnostics replace the blanket detection claim for eligible
  paired contrasts, with their assumptions stated explicitly.
- Missing and unparseable answers remain missing. Duplicated prompts, invalid
  age/SES reversals, and the custom opinion poll cannot support the original
  shared-construct interpretation.

See [the manuscript PDF](paper/manuscript.pdf), [LaTeX source](paper/main.tex),
[the analysis plan](docs/reanalysis_plan.md),
[changes to historical claims](docs/historical_claims.md), and
[the author handoff](docs/author_handoff.md).

## Reproduce the manuscript

Python 3.12.14 was used for the recorded run. A clean virtual environment is
recommended. Installing dependencies requires network access; all empirical
analysis after installation runs offline and makes no model/API calls.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python code/reanalyze.py
.venv/bin/python code/make_paper_assets.py
.venv/bin/python code/verify_revision.py
mkdir -p paper/build
tectonic paper/main.tex --outdir paper/build
```

The default analysis uses seed 20260927, 2,000 scenario bootstraps, 2,000
split-half partitions, and 20,000 permutations. It takes several minutes on a
desktop. Tectonic 0.17.0 was used to compile the PDF; its first run downloads TeX
resources. LaTeX sources and generated vector figures are also included for
other compatible TeX installations. The resulting PDF is `paper/build/main.pdf`.

`verify_revision.py` fails on stale input/code hashes and checks record counts,
pair completeness, score exports, rank probabilities, manuscript macros, and
concordance against an independent SciPy formula. The tests validate specific
implementation properties; they do not provide substantive human validation
of the probes or replace author review.

## Repository layout

| Path | Role |
| --- | --- |
| `data/raw/` | Original call records, unchanged |
| `data/*_items.json` | Original prompt/item banks, unchanged |
| `data/run_panel.csv` | Archived model labels, slugs, and catalog metadata |
| `code/measurement.py` | Revised parsing, scenario construction, and statistics |
| `code/reanalyze.py` | Offline analysis and all sensitivity specifications |
| `code/make_paper_assets.py` | Figures, tables, and manuscript result macros |
| `code/verify_revision.py` | Fail-closed checks on the revised release |
| `tests/` | Focused methodological and implementation tests |
| `results/revision/` | Response/scenario scores, coverage, statistics, provenance |
| `paper/` | Rebuilt LaTeX manuscript, bibliography, figures, generated tables |
| `docs/` | Retrospective plan, interpretation changes, and author handoff |

`results/revision/summary.json` records all inclusion sets, randomization settings,
software versions, and SHA-256 hashes of raw inputs and analysis code. The CSVs
retain the mapping from each scored response to its source record and scenario.
The archived labels “frontier” and “poscontrol” identify collection files; they
are not verified claims about current capability or positive-control validity.

## Historical material

The original `code/analyze.py`, `code/score_constructs.py`, canonical JSON files,
and design specification remain available for tracing the earlier analysis.
`code/check_consistency.py` checks only historical canonical quantities; it does
**not** validate the rebuilt manuscript. The revised statistics never read
historical canonical result files as empirical inputs. The old gender scorer is
used only in a clearly separated controlled resampling comparison.

`code/run_constructs.py` is an archived collection script. It requires paid API
access and can overwrite raw response files. It is unnecessary for reproduction
and is not invoked by the revised pipeline. Collection timestamps and resolved
provider versions are incomplete, so the archived corpus cannot be represented
as a fresh evaluation of current products.

## License

Code is released under the [MIT License](LICENSE). Existing data files are
released under [Creative Commons Attribution 4.0](LICENSE-data). Manuscript
authorship, declarations, and publication terms require the authors' review.
