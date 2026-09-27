# What changes in this revision

The baseline is repository commit `e77f9cae9fac90093cca6ce09efb08800c5de6c1`
and the earlier preprint [arXiv:2609.15995](https://arxiv.org/abs/2609.15995).
Raw response files, item banks, panel metadata, and canonical historical JSON
files are preserved. New estimates are in `results/revision/`.

| Earlier interpretation or procedure | What the released evidence supports |
| --- | --- |
| Ten benchmarks measure one common latent workplace-bias construct | Ten short, substantially adapted probe families measure different response properties. Source benchmark names do not confer validation. |
| Individual prompts can be split independently | Members of matched scenarios must remain together when evaluating stability across scenarios. The controlled historical comparison holds scoring and valid-response weights fixed. |
| Negative Spearman–Brown coefficients indicate negative reliability | Report raw rank correlation, defined partition counts, and partition ranges. The transformation is not a valid reliability interpretation when correlations are negative. |
| Absolute-score bootstrap intervals above zero demonstrate bias detection | Absolute noisy estimates can produce positive intervals under zero true effect. Eligible pair contrasts receive a separate conditional symmetry diagnostic with stated assumptions. |
| Every instrument's ranking is indistinguishable from chance | Revised absolute-net W = 0.230, p = 0.027; signed-net W = 0.114, p = 0.423. Neither a large p-value proves equivalence nor a small one validates a common construct. |
| BiasLab-independent agreement is established by the full comparison | Excluding the authors' mirrored-assertion family gives W = 0.189, p = 0.139. This dependence is reported, not hidden. |
| A response containing a choice token is that choice | Exact-answer parsing is primary. First-answer and anchored-wrapper parsers are sensitivity analyses. Refusals mentioning possible choices are not scored as decisions. |
| Incomplete pairs and unknown outputs can contribute as neutral | A matched repetition requires all members. Failed, empty, ambiguous, and unparseable outputs remain missing. |
| OpinionQA supplies a comparable bias magnitude | The released yes/no poll lacks human reference distributions and mixes policy, normative, and descriptive questions. It is descriptive only. |
| Twelve age/SES ambiguity IDs supply twelve scenarios | Each bank repeats a single prompt text twelve times; it supplies one unique scenario and is excluded from scenario-level inference. |
| Age/SES mirrored assertions replicate demographic swaps | Mechanical equality replacements and malformed reversals do not constitute symmetric swaps. These families are excluded from revised ranking inference. |
| Expanded panels directly test capability-range restriction | The additional models lack hiring and pronoun tasks. Shared-probe comparisons explicitly show changing complete-case model sets. |
| Native score signs support an overall regional bias direction | Stereotype congruence, group choice, ordinal agreement, and lexical contrasts have different meanings. No regional or cross-probe directional composite is reported. |

These are corrections and scope restrictions, not evidence that the original
full benchmarks are invalid. The revised main analysis and historical controlled
comparison answer different questions. In particular, complete pairing alone
raises historical decision stability toward zero, while the fully revised strict
decision analysis still has negative stability. The manuscript explains both.

No result has been “fixed” to achieve significance. All specified sensitivity
analyses, excluded families, and changing comparison panels are retained.
The plan is retrospective, with later implementation clarifications logged in
`reanalysis_plan.md`; it is not a preregistration.
