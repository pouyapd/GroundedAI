# Contributing

Thank you for helping. Small, focused pull requests with tests are easiest to review.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r backend/requirements/dev.txt
pip install --no-deps -e backend
cd frontend && npm ci
```

## Before opening a pull request

```bash
cd backend && ruff check groundedai tests && ruff format --check groundedai tests && mypy && pytest
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
```

* The default test suite must stay offline: no model downloads, no API keys, no network.
* Add a regression test for every bug fix.
* Do not change published benchmark reports by hand. Re-run the benchmark and add a new report
  file; describe the hardware and power source.
* Do not change files in `evaluation/datasets/groundedai-synthetic-v1/`. A dataset change needs
  a new dataset version (new directory), because reports refer to the dataset fingerprint.
* Never commit `.env` files, credentials, model weights, databases or uploaded documents.
* The held-out reliability test split (`evaluation/datasets/groundedai-reliability-v1/test/`
  and the other ignored folders listed in `.gitignore`) is private: it is not in the repository,
  and changes to the checks must not be tuned against it. Develop on the `dev` split.
* Changes to the claim or disagreement checks go into a new or opt-in version unless they fix a
  demonstrated bug; `claims-v1` and `disagreement-v1` reproduce the published reports.

## Labelling claims (human review)

`scripts/claim_review.py collect` exports real model answers from
`evaluation/datasets/groundedai-claims-review-v1` to a self-contained `review.html`. Label each
statement from the passages alone, without opening `checks.json`, download your labels file, and
add it to the review folder in a pull request. Two independent labellers per statement are
planned; see `evaluation/reviews/claims-review-v1/README.md`.

## Dependencies

Edit `backend/pyproject.toml`, then regenerate the locks (`backend/requirements/README.md`).
Every new dependency needs a clear purpose; add it to the table in `docs/ARCHITECTURE.md`.
