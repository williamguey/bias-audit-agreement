# Validation report

Recorded 27 September 2026 for the rebuilt manuscript.

- **Unit tests:** 20 passed (`python -m unittest discover -s tests -v`).
- **Full analysis:** completed with seed 20260927, 2,000 bootstraps,
  2,000 split partitions, and 20,000 permutations.
- **Input preservation:** all ten raw-data/item-bank/panel SHA-256 hashes match
  the bytes at baseline Git commit `e77f9cae9fac90093cca6ce09efb08800c5de6c1`.
- **Integrity checks:** 19,500 unique source records; score/status consistency;
  complete pairs; the full planned scenario grid including wholly absent units;
  coverage denominators; derived means; inclusion sets; tie shares; and code
  provenance all passed.
- **Concordance cross-check:** every main specification agrees with the
  tie-corrected Friedman-statistic conversion in SciPy, separately from the
  implementation used to calculate the reported W.
- **Manuscript checks:** all cited keys resolve; generated headline macros agree
  with analysis output; retrospective status and AI assistance are disclosed.
- **PDF:** 18 pages compiled with Tectonic 0.17.0. The final compilation has no
  warnings. Every page was rendered with Poppler and visually inspected for
  clipping, overlaps, broken glyphs, float placement, and table readability.
- **Repository hygiene:** `git diff --check` passed.

These checks concern reproducibility, implementation, and document production.
They are not an independent human assessment of construct validity, verification
by all authors, or a guarantee of publication. No new API data or human labels
were collected. See `author_handoff.md` for the remaining substantive review.
