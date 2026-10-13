# Verification records

- `BATTS_update_audit.json`: fork and upstream commits checked on 13 October 2026; the final run pins upstream 0.3.0.
- `BATTS_version_comparison.json`: all eight comparison tables and the null table match the preceding 0.2.2 run byte for byte; all Bayesian arrays match exactly.
- `preprocessing_validation.json`: independent reconstruction matched all original training/test values exactly.
- `generator_reproduction.json`: all four cleaned implementations reproduced selected CSVs byte for byte.
- `generator_validation.json`: input integrity, dimensions, composition constraints, and output checks.
- `generator_negative_checks.json`: validator rejection checks.
- `R_packages.csv`: versions installed for the final run; detailed session information is retained with results.

Python numerical dependencies and reporting dependencies are pinned in the
component requirements files. BATTS is pinned by source commit. Compiler,
platform and R-library changes can alter floating-point results; exact generator
replay was checked in the recorded CPU environments. Do not infer historical
neural-training reproducibility from these sample-replay checks.

`release_manifest.json` at the repository root binds every curated source,
input and result export. Run `python3 scripts/verify_release.py` after cloning.
Generated output directories are excluded from that release manifest.
