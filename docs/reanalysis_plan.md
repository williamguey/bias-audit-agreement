# Retrospective reanalysis specification

This is an exploratory, retrospective correction of the released study, not a
preregistration. The original manuscript, scripts, and preliminary concerns were
known before this document was written. It was written before running the new
pipeline. Original raw records and historical canonical values remain unchanged.

## Questions

1. Does treating each member of a matched contrast as an independent item change
   split-half rank stability relative to keeping the contrast intact?
2. How sensitive are model orderings to scoring, parsing, missingness, and the
   operational definition of the measured response?
3. What conclusions are supported by these short, adapted probes, without
   treating them as faithful implementations of their named source benchmarks?

## Fixed analysis decisions

- Use only the released responses. No new model calls, invented observations,
  retrospective human labels, or inferred missing responses.
- Record input hashes, package versions, source commit, seed, and all exclusions.
- Report attempted records, unique prompt texts, independent units, missing and
  unparseable outputs, and truncation separately. Repetitions and duplicate texts
  do not add independent scenarios.
- Parse constrained answers by an exact normalized enum. Reject prose, mixed
  answers, and ambiguous tokens rather than silently coding them as neutrality.
  Report legacy and permissive parsing only as labeled sensitivity analyses.
- Use an explicit pair manifest. All demographic variants, mirrored statements,
  or pronoun variants from one scenario belong to the same resampling unit.
  Average repetitions only after complete matched contrasts are formed.
- Deduplicate identical prompt texts for independent-unit counts and resampling.
- Treat generated gender cues as an unvalidated lexical operationalization;
  remove the non-gendered title "Dr" from male-name indicators. Lexicon scores
  remain proxies, not human-validated measures of discrimination.
- Keep each probe's interpretation separate. Stereotype congruence, male/female
  choice, attribute-word differences, and policy opinions are not interchangeable
  directions or a validated common latent construct. Do not average them into
  regional or overall model bias scores.
- Exclude the custom OpinionQA-labeled policy poll from bias-ranking inference:
  it lacks OpinionQA's human response distributions and a defensible common key.
- Exclude age/SES BiasLab-labeled items from inferential comparisons: their
  mechanically generated equality statements are not symmetric group swaps.
  Retain descriptive outputs and list the prompt defects.
- Exclude the age/SES BBQ-labeled probe from item-generalizing inference: each
  bank contains one unique prompt repeated under twelve IDs.
- Use raw split-half Spearman correlation as the primary stability statistic.
  Show the Spearman-Brown transform only where positive and explicitly note its
  assumptions; do not interpret negative transformed values as valid reliability.
- Compare item splits with block splits using the same scorers, rows, and model
  panel to isolate the effect of the resampling unit. Also report revised parsing
  and scoring separately, so changes are not attributed to pairing alone.
- For cross-probe comparison, report absolute net-score rankings and signed-score
  rankings as separate operational choices. Drop exactly constant vectors, not
  tools selected by a post-hoc standard-deviation cutoff. Report tool sets and
  complete-case model panels for every comparison.
- Main estimates use available complete units. Sensitivities require 80% unit
  coverage and exclude truncated responses. Missingness is never called fairness.
- Resample scenarios, preserving all model observations within a scenario.
  Bootstrap intervals describe scenario variation conditional on the fixed model
  panel. Rank uncertainty and tie-aware bottom-rank probabilities accompany means.
- Use tie-corrected Kendall W and a plus-one Monte Carlo permutation p value;
  non-significance is not equivalence or proof of universal ranking failure.
- Replace confidence intervals of absolute net scores as a detection test with
  a paired sign-flip reference only for genuine paired contrast probes. Flip a
  whole scenario jointly across models. Enumerate all patterns for <=16 units.
  This reference requires joint sign-exchangeability; demographic labels were not
  randomized, so it is a model-based symmetry diagnostic, not a causal test.
  Report absolute-score bootstraps as descriptive only for unpaired probes.
- Apply Holm adjustment within each panel/construct's paired symmetry diagnostics.
  Interpret lexicon findings within the declared proxy, not as established harm.
- Include simulations with known null/alternative structure to test the two
  statistical mechanisms. Label all simulation outputs synthetic.
- Treat the six additional models as a broader-panel sensitivity, not a proven
  capability positive control. Missing tools/items limit cross-panel comparison.

## Scope of the rebuilt paper

The revised paper is a measurement-pipeline case study based on the released
corpus. It must not claim that all ten original benchmarks fail, that all models
are biased, that non-significant concordance proves no meaningful agreement, or
that a preferred demographic response establishes unfair over-correction.

Independent validation of open-response coding and new faithful benchmark runs
would support a stronger future study. Their absence is a stated limitation,
not filled by generated annotations. All new choices and analyses are disclosed
as retrospective, and substantive AI assistance must be disclosed by the authors.

Implementation clarification after the low-budget smoke run: the expanded panel
must additionally be analyzed on the shared seven eligible probe families (and a
six-family sensitivity excluding mirrored assertions). Hiring and pronoun tasks
have no additional-model records; retaining them would silently discard the entire
additional panel. This is a design-coverage correction, not a result-based tool
selection. The all-tools complete-case result is retained and labeled explicitly.

Additional retrospective parsing sensitivity after inspecting short unparsed
answers: exact, unambiguous wrappers such as "I strongly disagree" and "I would
respond with: 'neutral'" are recognized by anchored grammar rules, alongside
the explicit-first-answer parser. These are reported as a separate sensitivity,
not used to revise the strict primary parser after seeing the results. This
check addresses the fact that format noncompliance can be semantically clear.

Final verification correction: the additional-model continuation file omits four
of the twelve bank scenarios for every model. The scenario export now retains
the full item-bank grid with these units marked missing, so coverage denominators
do not shrink to logged scenarios. This implements the planned missingness rule;
it does not impute outcomes. All statistics were rerun after this correction.
