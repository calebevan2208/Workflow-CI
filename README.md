# Workflow-CI

This folder contains example GitHub Actions workflow files and instructions for enabling CI for this repository.

Why this folder exists
- Keep a human-readable, reviewable copy of the CI workflow here before enabling it in `.github/workflows`.

How to enable
1. Copy `ci.yml` into `.github/workflows/ci.yml` (or create a symlink) to activate the workflow in GitHub Actions.

```bash
mkdir -p .github/workflows
cp Workflow-CI/ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Enable CI: add workflow from Workflow-CI"
git push
```

Notes
- The example workflow installs light dependencies to keep runs fast for PRs (TensorFlow is intentionally omitted in the default job).
- There is also an option to create a `workflow_dispatch` job for manual full-training runs.

If you want, I can:
- Copy `ci.yml` to `.github/workflows/ci.yml` and push the change for you (activate CI now) ✅
- Add a `pytest` scaffold and run tests in CI ✅
- Add an integration workflow for full training runs (manual dispatch) ✅

Tell me which action you prefer and I'll proceed.