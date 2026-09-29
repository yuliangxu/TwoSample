file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
# Rscript may encode spaces as ~+~ in its --file argument.
script_path <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
project_root <- normalizePath(file.path(script_dir, ".."))
suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
results_dir <- file.path(project_root, "output", "case_study")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
interval_path <- if (length(args)) normalizePath(gsub("~+~", " ", args[[1]], fixed = TRUE), mustWork = TRUE) else
  file.path(project_root, "data", "figure_S12", "test_pointwise_credible_intervals.csv")
if (!file.exists(interval_path)) stop("Missing interval table: ", interval_path)

results <- read.csv(interval_path)
stopifnot(all(is.finite(as.matrix(results[, c("posterior_mean", "ci_lower_025", "ci_upper_975")]))),
          all(results$ci_lower_025 <= results$ci_upper_975))
results$covers_zero <- results$ci_lower_025 <= 0 & results$ci_upper_975 >= 0

results <- results[order(results$posterior_mean), ]
results$rank <- seq_len(nrow(results))
results$interval_width <- results$ci_upper_975 - results$ci_lower_025
coverage_label <- sprintf(
  "%d of %d test observations' 95%% credible intervals include zero (%.1f%% coverage)",
  sum(results$covers_zero), nrow(results), 100 * mean(results$covers_zero)
)

if (all(results$covers_zero)) coverage_label <- sprintf(
  "All %d test observations' 95%% credible intervals include zero (100.0%% coverage)", nrow(results))
write.csv(results, file.path(results_dir, "figureS12_plot_data.csv"), row.names = FALSE)

theme_null <- theme_bw(base_size = 15) +
  theme(panel.grid = element_blank(), plot.tag = element_text(face = "bold"))

p_intervals <- ggplot(results, aes(rank, posterior_mean)) +
  geom_hline(yintercept = 0, linewidth = 0.8, color = "black") +
  geom_linerange(aes(ymin = ci_lower_025, ymax = ci_upper_975),
                 linewidth = 0.35, alpha = 0.35, color = "steelblue") +
  geom_point(size = 1.2, color = "darkred") +
  annotate("text", x = Inf, y = Inf, label = coverage_label,
           hjust = 1.03, vjust = 1.4, size = 4.1) +
  labs(x = "Test observations, ordered by posterior mean",
       y = "Log density ratio") +
  theme_null

output_path <- file.path(results_dir, "figureS12_ggplot2.pdf")
ggsave(output_path, p_intervals, width = 10, height = 5.2)
cat("Saved null diagnostic figure to", normalizePath(output_path), "\n")
