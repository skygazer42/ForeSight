# Pip Package Hardening (40 Tasks) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `foresight-ts` installable via `pip install ...` with robust package metadata, packaged resources
(paper metadata + small demo datasets), and CI coverage for building + installing wheels/sdists.

**Architecture:** Keep `src/foresight/` as the only installable package; treat Torch/Statsmodels as optional
extras; ship small resources under `foresight/data/` and access them via predictable paths.

**Tech Stack:** Python 3.10+, setuptools (PEP 621), ruff, pytest, optional PyTorch / statsmodels.

**Status:** Implemented on 2026-03-04 (see git diff for exact changes).

---

## Task List (40)

### Packaging metadata & versioning

1. **License metadata (setuptools deprecation)**
   - Modify: `pyproject.toml:1`
   - Change `project.license` to SPDX string and add `project.license-files`.
   - Verify: `python -m build --outdir /tmp/foresight_build_dist`

2. **Single source of truth for version**
   - Modify: `pyproject.toml:1`
   - Use `dynamic = ["version"]` + `tool.setuptools.dynamic.version.attr = "foresight.__version__"`.

3. **Sync `__version__` and packaging**
   - Modify: `src/foresight/__init__.py:1`
   - Ensure `__version__` matches intended release.

4. **Project URLs**
   - Modify: `pyproject.toml:1`
   - Add `project.urls` (Repository / Issues / Documentation) placeholders.

5. **Classifiers**
   - Modify: `pyproject.toml:1`
   - Add basic Trove classifiers (Programming Language, License, OS Independent).

6. **Dev/release tooling extras**
   - Modify: `pyproject.toml:1`
   - Add `build` and `twine` to `dev`, or add a new `release` extra.

7. **Package data rules**
   - Modify: `pyproject.toml:1`
   - Ensure `foresight/data/*.json` and `foresight/data/*.csv` are included in wheels.

8. **Wheel/sdist content guardrails**
   - Create: `MANIFEST.in`
   - Include: `README.md`, `LICENSE`, `docs/*.md` (optional)
   - Exclude: caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`) and temp build outputs.

9. **Expose package version via CLI**
   - Verify only (already present): `foresight --version`
   - File: `src/foresight/cli.py:659`

10. **Add `py.typed` (optional typing signal)**
   - Create: `src/foresight/py.typed`
   - Update package-data config to include it.

### Packaged resources (paper metadata + small datasets)

11. **Ship RNN paper metadata in the package**
   - Verify: `src/foresight/data/rnn_paper_metadata.json` is present in wheel.

12. **CLI loads paper metadata robustly**
   - Modify: `src/foresight/cli.py:15`
   - Prefer `foresight/data/rnn_paper_metadata.json` (installed), then `docs/` (dev), then env override.

13. **Bundle tiny demo datasets in the wheel**
   - Create: `src/foresight/data/catfish.csv`
   - Create: `src/foresight/data/ice_cream_interest.csv`

14. **Dataset registry supports packaged datasets**
   - Modify: `src/foresight/datasets/registry.py:1`
   - Add optional `package_rel_path` on `DatasetSpec` (or equivalent).

15. **Dataset loaders prefer packaged data**
   - Modify: `src/foresight/datasets/loaders.py:1`
   - If `package_rel_path` exists and no `data_dir`/env override is provided, load from package.

16. **CLI datasets path works for packaged datasets**
   - Modify: `src/foresight/cli.py:842`
   - Ensure `foresight datasets path catfish` prints a real path inside site-packages (wheel install).

17. **Tests: packaged dataset smoke**
   - Create: `tests/test_packaged_datasets_smoke.py`
   - Ensure `load_dataset("catfish")` works without `FORESIGHT_DATA_DIR`.

18. **Docs: explain packaged vs external datasets**
   - Modify: `README.md:1`
   - Mention that big datasets are not necessarily bundled; show `FORESIGHT_DATA_DIR` usage.

### Build + install smoke checks

19. **Add a build/install smoke script**
   - Create: `tools/smoke_build_install.py`
   - Build wheel+sdist to a temp dir and run import + `foresight --version`.

20. **Add a lightweight unit test for metadata presence**
   - Create: `tests/test_packaged_paper_metadata.py`
   - Import `foresight` and confirm `foresight/data/rnn_paper_metadata.json` exists on disk.

21. **CI: build wheel+sdist**
   - Modify: `.github/workflows/ci.yml:1`
   - Add step: `python -m pip install build` then `python -m build`.

22. **CI: install built wheel**
   - Modify: `.github/workflows/ci.yml:1`
   - After build: `pip install dist/*.whl` (non-editable path).

23. **CI: CLI smoke**
   - Modify: `.github/workflows/ci.yml:1`
   - Run: `foresight --version` and `foresight models list --prefix torch-rnnpaper --format json`.

24. **CI: pip dependency sanity**
   - Modify: `.github/workflows/ci.yml:1`
   - Run: `python -m pip check` after wheel install.

25. **CI: ensure no forbidden torch recurrent modules**
   - Verify only: already covered by `pytest`.

26. **Ruff: ensure tools are linted**
   - Verify: CI ruff step includes `tools/`.

27. **Docs generation reproducibility**
   - Verify only: `tests/test_docs_rnn_generated.py`

28. **Avoid relying on repo-relative paths at runtime**
   - Modify: `src/foresight/datasets/*` and `src/foresight/cli.py` as needed.

### Runtime ergonomics

29. **Improve dataset missing-file error message**
   - Modify: `src/foresight/datasets/loaders.py:18`
   - Mention `--data-dir` / `FORESIGHT_DATA_DIR`.

30. **Expose paper metadata in `foresight models info`**
   - Verify only: `tests/test_cli_models_info_paper_metadata.py`

31. **Keep Torch optional import boundaries**
   - Verify only: `tests/test_models_optional_deps_torch.py`

32. **Keep model registry deterministic**
   - Verify only: `tests/test_models_registry.py`

33. **Add `foresight.models` docs for new zoos**
   - Modify: `docs/rnn_paper_zoo.md:1`
   - Modify: `docs/rnn_zoo.md:1`

34. **Add an “install extras” guide**
   - Create: `docs/INSTALL.md`
   - Include: `pip install foresight-ts`, `pip install foresight-ts[torch]`, `pip install foresight-ts[all]`.

### Release workflow

35. **Add release checklist**
   - Create: `docs/RELEASE.md`
   - Include: bump version, regenerate docs, run tests, build, twine upload.

36. **Add a release sanity script**
   - Create: `tools/release_check.py`
   - Check: version consistency, docs generation, `python -m build`, `pytest -q`.

37. **Add `twine check` step to docs**
   - Modify: `docs/RELEASE.md:1`

38. **Document packaging build locally**
   - Modify: `docs/DEVELOPMENT.md:1`
   - Add: `python -m build`, wheel install smoke instructions.

39. **README: installation snippet**
   - Modify: `README.md:1`
   - Add: `pip install foresight-ts` and extras examples.

40. **Final verification pass**
   - Run: `ruff check src tests tools`
   - Run: `ruff format --check src tests tools`
   - Run: `pytest -q`
   - Run: `python -m build --outdir /tmp/foresight_build_dist`
