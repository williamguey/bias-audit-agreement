# Bias Audits Detect Bias but Disagree on Ranking

Replication data and code for *"Bias Audits Detect Bias but Disagree on
Ranking: Evidence from Ten Instruments and Ten Frontier Models."*

Ten extrinsic bias-audit instruments are run over one shared panel of ten
frontier models through a single pooled inference gateway, on three workplace
constructs (gender, age, socioeconomic status). The tools reliably **detect**
bias but their model **rankings** are statistically indistinguishable from
chance. This repository lets you reproduce every number reported in the paper
from the raw model responses, with no API calls.

## Layout

```
code/
  run_constructs.py     regenerate raw responses (needs an OpenRouter key)
  score_constructs.py   raw responses -> per-model item scores (age, SES)
  analyze.py            raw responses -> gender statistics and reliability
  check_consistency.py  recompute every headline number from raw data and
                        assert it matches the paper's reported values
data/
  raw/                  primary data: one file per construct / panel
    gender_core_raw.json    gender, original five tools
    gender_added_raw.json   gender, five added tools
    poscontrol_raw.json     positive control, six weaker models
    age_raw.json            age construct
    ses_raw.json            socioeconomic-status construct
  *_items.json          item banks
  run_panel.csv         the ten-model panel (slugs, provenance, flags)
  *_scores.json         derived per-model scores
  canonical_values.json      single source of truth for every paper number
  construct_canonical.json   age/SES canonical results
  harmonization_spec.md      how each tool is mapped to a common score
```

`data/raw/` holds the primary data. Everything else under `data/`
(`*_scores.json`, the canonical files) is derived from it by the scripts in
`code/` and is committed for convenience. The LaTeX manuscript and figures
are not part of this repository; it contains only what is needed to
reproduce the study.

## Reproduce the analysis (no network, ~1 min)

```
pip install -r requirements.txt
python code/check_consistency.py     # recompute all numbers, verify vs. paper
python code/analyze.py               # gender statistics + reliability
python code/score_constructs.py      # rebuild age/SES scores from raw
```

`check_consistency.py` is the integrity gate: it reruns the tie-corrected
Kendall's W ranking pipeline on the raw response files and asserts that the
paper's construct results (age W=0.32, p=0.012; SES W=0.10; the socioeconomic
hiring count of 273 of 278) reproduce from data rather than from any stored
draft.

## Re-run the audit from scratch (needs an OpenRouter key)

```
export OPENROUTER_KEY=sk-or-...
python code/run_constructs.py        # writes data/raw/*_raw.json
```

All models are reached through one pooled routing gateway at temperature 0
with three repetitions per item.

## Panel

- **US:** GPT-5.2, Claude Sonnet 4, Gemini 3 Flash, Llama 4 Maverick
- **China:** Qwen3-235B, DeepSeek V3.2, GLM-4.7, MiMo V2.5, Kimi K2
- **EU:** Mistral Large

## License

Code is released under the MIT License (`LICENSE`). The data files are
released under Creative Commons Attribution 4.0 (`LICENSE-data`).
