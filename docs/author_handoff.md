# Author handoff — rebuilt bias-audit manuscript

Prepared 27 September 2026. This is a substantive reconstruction from the
released repository, not a recovery of the missing original LaTeX folder.

## Recommendation

Use **From Bias Scores to Model Rankings: A Measurement Audit of Ten LLM Probe
Families** as the working title. I would not submit the previous title and
blanket conclusions unchanged. The strongest defensible contribution is an
auditable case study showing how measurement and analysis choices alter
conclusions from the same response corpus.

The current draft is a more defensible submission candidate, but it still needs
the authors' substantive review. Short custom banks, unvalidated lexical proxies,
incomplete collection metadata, and retrospective analysis limit what this study
can establish. Better code does not cure those design limitations.

My suggested first journal route is **Data & Policy**, framed around the quality
of evidence used for model selection and assurance. Its stated interests include
uncertainty, fairness, reliability, and validation; it accepts research articles
and permits disclosed AI assistance. This is a fit assessment, not a prediction
of acceptance. The draft includes a policy-significance paragraph and substantive
AI disclosure. Before submission, align the final version with its
[official author instructions](https://www.cambridge.org/core/journals/data-and-policy/information/author-instructions/preparing-your-materials),
including declarations, corresponding-author details, and any caption disclosures.

Do not treat this AI-drafted version as automatically eligible for FAccT 2027.
The current [author guide](https://facctconference.org/2027/authorguide.html)
contains broad disclosure language but also says that it prohibits LLM-generated
publication text. That conflict needs resolution if choosing that conference.
Do not conceal the assistance or assume that deleting the disclosure solves it.

## What was rebuilt

- A complete LaTeX manuscript with methods, related work, results, discussion,
  limitations, ethics, AI disclosure, bibliography, and methodological appendices.
- Four figures and six result tables, generated directly from analysis outputs.
- An offline pipeline starting from all 19,500 raw call records, without fresh
  model calls, fabricated observations, or generated “human” annotations.
- A controlled historical comparison that changes the resampling unit while
  holding the original scoring rules and response weights fixed.
- Explicit parsing and missingness, complete matched repetitions, unique
  scenarios, tie-corrected concordance, uncertainty estimates, conditional
  symmetry diagnostics, and declared sensitivity specifications.
- A retrospective analysis plan, input and code hashes, pinned core dependencies,
  20 tests, and fail-closed verification against an independent SciPy formula.
- Original raw records, item banks, panel metadata, and historical canonical
  values preserved. Historical documentation is clearly separated from the
  revised interpretation.

The response and scenario exports allow reviewers to trace results to source
records. `results/revision/summary.json` records the exact comparison panels,
settings, package versions, and input/code checksums.

## The conclusions changed

| Quantity | Revised result and interpretation |
| --- | --- |
| Historical decision stability: item vs scenario split | r = −0.495 vs 0.048, under fixed historical scoring. Pairing changes the diagnosis but does not establish dependable ranking. |
| Primary absolute-net concordance | W = 0.230, p = 0.027, nine models and nine probes. A modest exploratory concordance result. |
| Signed-net concordance | W = 0.114, p = 0.423. Native directions do not share a validated demographic axis. |
| Excluding mirrored assertions | W = 0.189, p = 0.139. Dependence on the authors' BiasLab-derived family is visible. |
| Detection | Positive intervals on absolute estimated scores are insufficient. Conditional paired diagnostics support only specific operational statements under explicit symmetry assumptions. |
| Age/SES extensions | Duplicated ambiguity prompts and invalid assertion reversals prevent treating these as clean replications of a shared design. |
| Additional models | Hiring/pronoun tasks are absent; continuation data also omit bank scenarios. Shared-task panels and planned-unit coverage are explicit. |

See `docs/historical_claims.md` for the complete correction map. The manuscript
does not claim universal failure of the source benchmarks, prove an absence of
bias, or identify a universally fairest model.

## Author work before submission

1. **Independently verify the substance.** Have an author check the parsing rules,
   scenario mapping, score interpretations, inference assumptions, and the
   differences from the earlier preprint. Reproduce the pipeline and inspect
   several raw-to-score examples for every retained family. Passing tests alone
   is not author verification.
2. **Confirm manuscript metadata.** The author list and affiliations were carried
   from the previous manuscript and need confirmation. Supply the corresponding
   author, ORCID, funding, competing interests, and author contributions from
   actual records; obtain all coauthors' agreement to the revised conclusions.
   No claims of approval or absence of funding were invented.
3. **Decide the empirical scope.** For this narrow measurement case study, retain
   the limitations. To argue about broad benchmark validity, actual social harm,
   or deployable model selection, collect additional evidence first. The highest
   value additions would be independent human checks of open-response scoring
   and a faithfully implemented external benchmark with fuller scenario coverage.
   Those additions have not been performed.
4. **Finalize the venue version.** Keep the substantive AI disclosure; adapt the
   title page, acknowledgements, declarations, and format to the chosen journal.
   The supplied cover letter is a draft and has not been sent.
5. **Update the preprint transparently after author review.** Preserve the earlier
   version and clearly explain the changed results. The new manuscript already
   identifies the previous preprint rather than presenting its reanalysis as
   unrelated new data.

No submission, preprint replacement, or coauthor communication was performed.

## Reproduction

From the source-package root, follow `README.md`. The pinned analysis environment
used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.18.1, and Matplotlib 3.11.2;
the PDF was compiled with Tectonic 0.17.0. The full analysis uses 2,000 bootstrap
samples, 2,000 partitions, and 20,000 permutations with seed 20260927.
