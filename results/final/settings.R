list(batts_repository = "nawaya040/BATTS", batts_commit = "147e511d211919afefb8decac6333e66d24919db", 
    batts_version = "0.3.0", keys = c("icfm", "dt", "d", "mbgan"
    ), labels = c(icfm = "ICFM", dt = "DT", d = "D", mbgan = "MBGAN"
    ), dimensions = c(train = 1166L, test = 292L, generated = 1000L, 
    taxa = 123L), seed = 1L, bayes = list(num_trees = 200L, max_resol = 0L, 
        learn_rate = 0.01, n_bins = 100L, n_min_obs_per_node = 1L, 
        n_ratio_per_node = 1e-100, margin_scale = 0.1, use_gradient = FALSE, 
        size_burnin = 2000L, size_backfitting = 1000L, thin = 1L, 
        prob_moves = c(0.333333333333333, 0.333333333333333, 
        0.333333333333333), lambda_0 = 5, omega_prior_parameters = c(1, 
        1), tree_priors = c(0.95, 2), output_BART_ensembles = FALSE, 
        quiet = TRUE), boosting = list(num_trees_max = 1000L, 
        K_CV = 5L, max_resol = 4L, learn_rate = 0.01, n_bins = 32L, 
        n_min_obs_per_node = 1L, n_ratio_per_node = 1e-100, margin_scale = 0.1, 
        quiet = TRUE, subsample_fraction = 1), null_seed = 2026L, 
    figures = c("6", "7", "S8", "S9", "S10", "S11", "S12"), null = list(
        num_trees = 200L, max_resol = 0L, learn_rate = 0.01, 
        n_bins = 100L, n_min_obs_per_node = 1L, n_ratio_per_node = 1e-100, 
        margin_scale = -1, use_gradient = FALSE, size_burnin = 500L, 
        size_backfitting = 500L, thin = 1L, prob_moves = c(0.333333333333333, 
        0.333333333333333, 0.333333333333333), lambda_0 = 5, 
        omega_prior_parameters = c(1, 1), tree_priors = c(0.95, 
        2), output_BART_ensembles = TRUE, quiet = TRUE))
