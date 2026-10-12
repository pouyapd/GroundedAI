# Evaluation reports

* `published/`: reports referenced by the documentation and shown in the dashboard. Each run
  writes three files with the same id: `.json` (full report, including per-question rows),
  `.md` (summary) and `.csv` (per-question rows). They are measured outputs; do not edit them.
* `local/`: scratch output of local runs (git-ignored). Copy a report to `published/` only
  when it should become part of the record.

Every report records the dataset name, version and fingerprint, the configuration, the
hardware, the power source, the package versions and a list of limitations.

The reports published on 2026-10-11 and 2026-10-12 record the dataset fingerprint
`5a04e91e09086368`, computed before the fingerprint ignored line endings; the same dataset now
gives `fa56ebd64e3b1eee`. See `docs/EVALUATION.md` ("Dataset").
