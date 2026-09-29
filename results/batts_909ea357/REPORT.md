# Case-study rerun with BATTS 909ea357

BATTS source commit: `909ea357bdb3f97018613552ef13430d1bb60c6e`.

Eight real-versus-generated comparisons were refitted with gradient boosting,
Hellinger boosting, and Bayesian additive trees. The null experiment was
refitted separately, and Figures 6, 7, S8, S9, S10, S11, and S12 were redrawn.
Real and generated input compositions are unchanged. Figures 6 and S8 depend
only on these compositions, so their ordinations are unaffected by BATTS.

## Posterior means compared with the saved paper results

Generator keys: d = Dirichlet; dt = Dirichlet Tree; icfm = ICFM; mbgan = MB-GAN.

Differences below are in log-density-ratio units and include real and
generated observations. They describe the rerun versus the bundled results;
they do not isolate individual package changes from numerical/environment effects.

| Split | Generator | Maximum absolute difference | RMSE | Correlation |
| --- | --- | ---: | ---: | ---: |
| train | d | 8.3565 | 1.71036 | 0.991811 |
| train | dt | 6.85644 | 1.84006 | 0.991573 |
| train | icfm | 9.37271 | 4.46587 | 0.978792 |
| train | mbgan | 3.24597 | 0.8422 | 0.833963 |
| test | d | 1.62195 | 0.478956 | 0.913285 |
| test | dt | 1.60627 | 0.526027 | 0.979928 |
| test | icfm | 4.23893 | 1.08362 | 0.948311 |
| test | mbgan | 0.289039 | 0.17451 | 0.360088 |

## Pointwise intervals containing zero

These are descriptive counts for the fitted comparisons, not coverage
estimates under a known null. Only the separate null experiment has that design.

| Split | Generator | Observations | n | Paper | Rerun | Changed interval class |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| train | d | Real | 1166 |    0 |    0 |   0 |
| train | d | Generated | 1000 |    0 |    0 |   0 |
| train | dt | Real | 1166 |    0 |    0 |   0 |
| train | dt | Generated | 1000 |    0 |    0 |   0 |
| train | icfm | Real | 1166 |    3 |    0 |   3 |
| train | icfm | Generated | 1000 |   29 |    0 |  33 |
| train | mbgan | Real | 1166 |  304 |  335 | 349 |
| train | mbgan | Generated | 1000 |  274 |  293 | 279 |
| test | d | Real |  292 |   75 |   82 |  23 |
| test | d | Generated | 1000 |  278 |  303 | 207 |
| test | dt | Real |  292 |   22 |   25 |   7 |
| test | dt | Generated | 1000 |   52 |   41 |  31 |
| test | icfm | Real |  292 |    2 |    3 |   3 |
| test | icfm | Generated | 1000 |  103 |   57 |  97 |
| test | mbgan | Real |  292 |  292 |  292 |   0 |
| test | mbgan | Generated | 1000 | 1000 | 1000 |   0 |

## Held-out null experiment

292 of 292 test intervals contain zero (100.0% observed coverage).

The maximum change in the posterior mean from the paper is 0.0402331.

## Settings and scope

Comparisons: seed 1 for each estimator; 200 Bayesian trees; 2,000 burn-in
iterations; 1,000 retained iterations; thinning 1; fixed lambda 5; margin 0.1.
Boosting: five-fold CV; up to 1,000 trees; depth 4; learning rate 0.01; 32 bins.
Training and testing comparisons are fitted separately, following the original
case-study scripts; these are not predictions from a model trained on the other split.

Null: seed 2026; balanced random labels for the real training samples; 200 trees;
500 burn-in and 500 retained iterations; fixed lambda 5; margin -1;
evaluation on all 292 held-out real test observations.

8 of 16 boosting fits selected the 1,000-tree search limit; the original search range was retained.

This reproduces the specified single-chain workflow; no new multi-chain
convergence claim is made. The four generative models were not retrained.

## Files

- `revision/train/` and `revision/test/`: all ten-column estimator/quantile tables.
- `comparison_to_paper.csv`: differences for all estimators and posterior summaries.
- `interval_summary.csv`: interval classifications by split, generator and source.
- `boosting_cv_summary.csv`: CV-selected tree counts and losses.
- `null/`: held-out intervals and coverage; `null_comparison_to_paper.csv`: numerical changes.
- `input_checksums.csv`, `BATTS_commit.txt`, `settings.R`, `sessionInfo.txt`: run provenance.
- `fits/` in the local output directory: cached fitted objects and posterior draws.

## Figures

- [Figure 6](figures/figure6_ggplot2.pdf)
- [Figure 7](figures/figure7_ggplot2.pdf)
- [Figure S8](figures/figureS8_ggplot2.pdf)
- [Figure S9](figures/figureS9_ggplot2.pdf)
- [Figure S10](figures/figureS10_ggplot2.pdf)
- [Figure S11](figures/figureS11_ggplot2.pdf)
- [Figure S12](figures/figureS12_ggplot2.pdf)

Use `Rscript scripts/reproduce_case_study_batts.R 2` from the repository root
after installing the exact BATTS commit specified in the main README.
