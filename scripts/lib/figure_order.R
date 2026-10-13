# Presentation order; the recorded fitting sequence remains unchanged.
figure_keys <- c("d", "dt", "icfm", "mbgan")
figure_labels <- case_study$labels[figure_keys]
stopifnot(setequal(figure_keys, case_study$keys))
