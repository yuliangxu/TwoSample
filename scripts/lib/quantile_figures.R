suppressPackageStartupMessages({ library(ggplot2); library(vegan); library(patchwork); library(scales) })
data_dir <- file.path(project_root, "data")
ratio_root <- Sys.getenv("TWO_SAMPLE_RATIO_ROOT", file.path(project_root, "data", "revision"))
output_dir <- Sys.getenv("TWO_SAMPLE_FIGURE_DIR", file.path(project_root, "output", "case_study"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
cache_dir <- file.path(output_dir, "cache")
dir.create(cache_dir, showWarnings = FALSE)
# Fix eigenvector signs to the manuscript orientation. The farthest point
# on each axis is a reproducible sign anchor; distances are unchanged.
axis_signs <- list(
  train = list(d = c(-1, -1), dt = c(1, -1), icfm = c(-1, -1), mbgan = c(-1, 1)),
  test = list(d = c(-1, 1), dt = c(1, 1), icfm = c(-1, 1), mbgan = c(1, 1)))
keys <- c("d", "dt", "icfm", "mbgan")
method_labels <- c(d = "Dirichlet", dt = "Dirichlet Tree", icfm = "ICFM", mbgan = "MB-GAN")
# Matplotlib SymLogNorm(base=10, linscale=1), as in the original Python plots.
symlog <- function(threshold) {
  a <- 1 / (1 - 10^-1)
  scales::new_transform("symlog10",
    transform = function(x) sign(x) * ifelse(abs(x) <= threshold,
      abs(x) * a, threshold * (a + log10(pmax(abs(x), threshold) / threshold))),
    inverse = function(y) sign(y) * ifelse(abs(y) <= threshold * a,
      abs(y) / a, threshold * 10^(abs(y) / threshold - a)))
}
for (number in figures) {
  stopifnot(number %in% c("S10", "S11"))
  split <- if (number == "S10") "train" else "test"
  real <- as.matrix(read.csv(file.path(data_dir, paste0("sample_", split, ".csv")), header = FALSE))
  panels <- lapply(keys, function(key) {
    generated <- as.matrix(read.csv(file.path(data_dir, paste0("sample_", key, ".csv")), header = FALSE))
    # Classical Bray-Curtis PCoA without additive correction, matching
    # utils/microbiome_help.py::compute_pcoa_coords used for S10/S11.
    message("Preparing classical PCoA: ", split, "/", key)
    inputs <- file.path(data_dir, paste0("sample_", c(split, key), ".csv"))
    fingerprint <- list(inputs = unname(tools::md5sum(inputs)), algorithm = "bray_cmdscale_add_false_v1")
    cache_path <- file.path(cache_dir, paste0("classical_", split, "_", key, ".rds"))
    cached <- if (file.exists(cache_path)) readRDS(cache_path) else NULL
    if (!is.null(cached) && identical(cached$fingerprint, fingerprint)) {
      points <- cached$points
    } else {
      points <- cmdscale(vegdist(rbind(real, generated), method = "bray"), k = 2, eig = TRUE, add = FALSE)$points
      saveRDS(list(fingerprint = fingerprint, points = points), cache_path)
    }
    for (axis in 1:2) {
      anchor <- which.max(abs(points[, axis]))
      points[, axis] <- points[, axis] * sign(points[anchor, axis]) * axis_signs[[split]][[key]][axis]
    }
    z <- as.matrix(read.csv(file.path(ratio_root, split, paste0("log_w_", key, ".csv")), header = FALSE))
    stopifnot(nrow(z) == nrow(points), ncol(z) >= 10)
    data.frame(PC1 = points[, 1], PC2 = points[, 2],
      source = rep(c("Real", "Generated"), c(nrow(real), nrow(generated))),
      method = method_labels[[key]], key = key, mean = z[, 3], lower = z[, 4], upper = z[, 10])
  })
  df <- do.call(rbind, panels)
  df$method <- factor(df$method, levels = unname(method_labels))
  df$source <- factor(df$source, levels = c("Generated", "Real"))
  # Match the fixed limits used for the manuscript's train/test grids.
  limit <- if (split == "train") 30 else 12
  titles <- c(mean = "Posterior mean of DRE", lower = "Lower 2.5% MCMC quantile of DRE",
              upper = "Upper 97.5% MCMC quantile of DRE")
  plots <- lapply(names(titles), function(quantity) {
    ggplot(df, aes(PC1, PC2, color = .data[[quantity]], shape = source)) +
      geom_point(size = 0.65, alpha = 0.75, stroke = 0.3) +
      facet_wrap(~method, nrow = 1, scales = "free") +
      scale_shape_manual(values = c(Generated = 4, Real = 16)) +
      scale_color_gradientn(colors = c("#00004D", "blue", "white", "red", "#800000"),
        limits = c(-limit, limit), oob = squish, transform = symlog(limit * 0.01),
        breaks = c(-10, -1, -0.1, 0, 0.1, 1, 10),
        labels = c(expression(-10^1), expression(-10^0), expression(-10^-1), "0",
                   expression(10^-1), expression(10^0), expression(10^1))) +
      labs(x = "PCoA 1", y = "PCoA 2", color = "log(w)", shape = NULL,
           title = paste0(if (split == "train") "Train" else "Test", " vs. Generated: ", titles[[quantity]])) +
      guides(color = guide_colorbar(position = "right", barheight = grid::unit(52, "mm")),
             shape = guide_legend(position = "bottom", override.aes = list(color = "#222222", size = 2, alpha = 1))) +
      theme_bw(base_size = 11) + theme(panel.grid = element_blank(),
        strip.background = element_rect(fill = "white"), strip.text = element_text(face = "bold"),
        plot.title = element_text(hjust = 0.5), legend.text = element_text(size = 8),
        legend.box = "vertical")
  })
  combined <- wrap_plots(plots, ncol = 1)
  ggsave(file.path(output_dir, paste0("figure", number, "_ggplot2.pdf")), combined, width = 12, height = 10)
  write.csv(df, file.path(output_dir, paste0("figure", number, "_plot_data.csv")), row.names = FALSE)
  cat("Saved Figure", number, "\n")
}
