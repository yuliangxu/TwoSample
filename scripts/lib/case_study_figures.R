suppressPackageStartupMessages({ library(ggplot2); library(vegan); library(patchwork) })

data_dir <- file.path(project_root, "data")
ratio_root <- file.path(project_root, "data", "revision")
output_dir <- file.path(project_root, "output", "case_study")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

keys <- c("d", "dt", "icfm", "mbgan")
method_labels <- c(d = "Dirichlet", dt = "Dirichlet Tree", icfm = "ICFM", mbgan = "MB-GAN")
real <- list(train = as.matrix(read.csv(file.path(data_dir, "sample_train.csv"), header = FALSE)),
             test = as.matrix(read.csv(file.path(data_dir, "sample_test.csv"), header = FALSE)))
generated <- setNames(lapply(keys, function(k) as.matrix(read.csv(file.path(data_dir, paste0("sample_", k, ".csv")), header = FALSE))), keys)

theme_jasa <- theme_bw(base_size = 16) +
  theme(panel.grid = element_blank(), strip.background = element_rect(fill = "white"),
        strip.text = element_text(face = "bold", size = 16),
        legend.text = element_text(size = 14), legend.title = element_text(size = 15),
        legend.position = "bottom")
ratio_colors <- c("darkblue", "blue", "deepskyblue", "cyan", "white", "#FFFF99", "yellow", "orange", "red", "darkred")

pcoa <- function(x) {
  fit <- cmdscale(vegdist(x, method = "bray"), k = 2, eig = TRUE, add = TRUE)
  positive <- fit$eig[fit$eig > 0]
  pct <- 100 * fit$eig[1:2] / sum(positive)
  list(points = fit$points[, 1:2, drop = FALSE], pct = pct)
}

figure_splits <- c("6" = "test", "7" = "test", "S8" = "train", "S9" = "train")
stopifnot(all(figures %in% names(figure_splits)))
cache_dir <- file.path(output_dir, "cache")
dir.create(cache_dir, showWarnings = FALSE)
ordination_path <- file.path(output_dir, "pcoa_coordinates.rds")
# Cache coordinates only when input content and the ordination algorithm match.
{
  ordination <- list()
  for (split in unique(figure_splits[figures])) for (key in keys) {
    message("Preparing PCoA: ", split, "/", key)
    n_real <- nrow(real[[split]])
    inputs <- file.path(data_dir, paste0("sample_", c(split, key), ".csv"))
    fingerprint <- list(inputs = unname(tools::md5sum(inputs)), algorithm = "bray_cmdscale_add_true_v1")
    cache_path <- file.path(cache_dir, paste0(split, "_", key, ".rds"))
    cached <- if (file.exists(cache_path)) readRDS(cache_path) else NULL
    if (!is.null(cached) && identical(cached$fingerprint, fingerprint)) {
      ord <- cached$ordination
    } else {
      ord <- pcoa(rbind(real[[split]], generated[[key]]))
      saveRDS(list(fingerprint = fingerprint, ordination = ord), cache_path)
    }
    ordination[[paste(split, key, sep = "_")]] <- data.frame(
      PC1 = ord$points[, 1], PC2 = ord$points[, 2],
      sample = rep(c("Real", "Generated"), c(n_real, nrow(generated[[key]]))),
      split = split, method = method_labels[[key]], key = key,
      pct1 = ord$pct[1], pct2 = ord$pct[2]
    )
  }
  ord_df <- do.call(rbind, ordination)
  saveRDS(ord_df, ordination_path)
}
ord_df$method <- factor(ord_df$method, levels = unname(method_labels))

make_pcoa_figure <- function(split, number) {
  plot_data <- ord_df[ord_df$split == split, ]
  plot_data$method <- factor(plot_data$method, levels = unname(method_labels))
  p <- ggplot(plot_data, aes(PC1, PC2, color = sample, shape = sample)) +
    geom_point(size = 1.05, alpha = 0.6) +
    scale_color_manual(values = c(Real = "#2166AC", Generated = "#E66101")) +
    scale_shape_manual(values = c(Real = 16, Generated = 4)) +
    facet_wrap(~method, nrow = 1, scales = "free") +
    labs(x = "PCoA 1", y = "PCoA 2", color = NULL, shape = NULL) + theme_jasa +
    guides(
      color = guide_legend(override.aes = list(
        size = 4.5, alpha = 1, shape = c(4, 16), color = c("#E66101", "#2166AC")
      )),
      shape = "none"
    ) +
    theme(legend.key.width = unit(8, "mm"), legend.spacing.x = unit(3, "mm"))
  ggsave(file.path(output_dir, paste0("figure", number, "_ggplot2.pdf")),
         p, width = 14, height = 5.2)
  write.csv(plot_data, file.path(output_dir, paste0("figure", number, "_plot_data.csv")), row.names = FALSE)
}


load_ratio <- function(split, key) {
  z <- as.matrix(read.csv(file.path(ratio_root, split, paste0("log_w_", key, ".csv")), header = FALSE))
  n_real <- nrow(real[[split]])
  if (ncol(z) < 10L || nrow(z) != n_real + nrow(generated[[key]]))
    stop("Sample/posterior dimension mismatch for ", split, "/", key)
  data.frame(row = seq_len(nrow(z)), source = rep(c("Real", "Generated"), c(n_real, nrow(z) - n_real)),
             mean = z[, 3], lower = z[, 4], upper = z[, 10], key = key,
             method = method_labels[[key]], split = split)
}

theme_result <- theme_bw(base_size = 9.5) +
  theme(
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "white"),
    strip.text = element_text(face = "bold", size = 8.3, margin = margin(t = 1.5, b = 1.5)),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 8.5),
    legend.text = element_text(size = 8.5),
    legend.title = element_text(size = 9),
    plot.tag = element_text(face = "bold", size = 10)
  )

make_result_figure <- function(split, number) {
  ratios <- do.call(rbind, lapply(keys, function(k) load_ratio(split, k)))
  panels <- lapply(keys, function(k) {
    coords <- ordination[[paste(split, k, sep = "_")]]
    z <- ratios[ratios$key == k, ]
    cbind(z, PC1 = coords$PC1, PC2 = coords$PC2)
  })
  df <- do.call(rbind, panels)
  df$method <- factor(df$method, levels = unname(method_labels))
  df$ci_class <- factor(ifelse(df$upper < 0, "Below 0", ifelse(df$lower > 0, "Above 0", "Covers 0")),
                        levels = c("Below 0", "Covers 0", "Above 0"))
  limit <- max(abs(df$mean), na.rm = TRUE)
  p_mean <- ggplot(df, aes(PC1, PC2, color = mean)) +
    geom_point(size = 0.72, alpha = 0.68) +
    facet_wrap(~method, nrow = 1, scales = "free") +
    scale_color_gradientn(
      colors = ratio_colors, limits = c(-limit, limit),
      values = scales::rescale(c(-limit, 0, limit)),
      guide = guide_colorbar(direction = "horizontal", barheight = unit(3.5, "mm"),
                             barwidth = unit(45, "mm"), title.position = "left",
                             title.vjust = 0.5)
    ) +
    labs(x = NULL, y = "PCoA 2", color = "Posterior mean log density ratio", tag = "(a)") +
    theme_result +
    theme(legend.position = "bottom",
          axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          legend.box.margin = margin(t = -4, r = 0, b = -4, l = 0),
          plot.margin = margin(t = 3, r = 4, b = 2, l = 4))

  p_class <- ggplot(df, aes(PC1, PC2, color = ci_class, alpha = ci_class)) +
    geom_point(size = 0.72) +
    facet_wrap(~method, nrow = 1, scales = "free") +
    scale_color_manual(
      values = c("Below 0" = "blue", "Covers 0" = "#333333", "Above 0" = "red"),
      breaks = c("Above 0", "Covers 0", "Below 0"),
      guide = guide_legend(direction = "horizontal", nrow = 1,
                           override.aes = list(shape = 15, size = 6, alpha = 1),
                           title.position = "left", title.vjust = 0.5)
    ) +
    scale_alpha_manual(values = c("Below 0" = 0.28, "Covers 0" = 0.88, "Above 0" = 0.28),
                       guide = "none") +
    labs(x = "PCoA 1", y = "PCoA 2", color = "95% interval", tag = "(b)") +
    theme_result +
    theme(legend.position = "bottom",
          legend.box.margin = margin(t = -4, r = 0, b = -4, l = 0),
          plot.margin = margin(t = 2, r = 4, b = 2, l = 4))

  gen <- ratios[ratios$source == "Generated", ]
  score <- aggregate(abs(mean) ~ row, gen, mean)
  chosen <- head(score$row[order(score[[2]], decreasing = TRUE)], 20)
  ci_df <- gen[gen$row %in% chosen, ]
  ci_df$display_row <- factor(ci_df$row, levels = rev(chosen))
  p_ci <- ggplot(ci_df, aes(display_row, mean, color = method)) +
    geom_hline(yintercept = 0, color = "grey40") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2,
      position = position_dodge(width = 0.7)) +
    geom_point(position = position_dodge(width = 0.7), size = 1.5) +
    labs(x = "Generated-sample row", y = "Log density ratio (95% CI)",
         color = NULL, tag = "(c)") +
    theme_result +
    theme(legend.position = "bottom", legend.key.height = unit(4.5, "mm"),
          legend.key.width = unit(4.5, "mm"), legend.text = element_text(size = 8.5),
          legend.spacing.x = unit(1.5, "mm"),
          legend.box.margin = margin(t = -3, r = 0, b = -4, l = 0),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 7.5),
          plot.margin = margin(t = 3, r = 4, b = 2, l = 4)) +
    guides(color = guide_legend(nrow = 1, byrow = TRUE,
                                override.aes = list(size = 3)))
  combined <- (p_mean / p_class / p_ci) +
    plot_layout(heights = c(0.9, 1, 0.72))
  prefix <- paste0("figure", number, "_ggplot2")
  # Match the 6.5-inch text width from letter paper with 1-inch margins so
  # LaTeX can include the PDF at natural size without shrinking its fonts.
  ggsave(file.path(output_dir, paste0(prefix, ".pdf")), combined, width = 6.5, height = 6.5)
  write.csv(df, file.path(output_dir, paste0("figure", number, "_plot_data.csv")), row.names = FALSE)
}

for (number in figures) {
  split <- figure_splits[[number]]
  if (number %in% c("6", "S8")) make_pcoa_figure(split, number)
  else make_result_figure(split, number)
}
cat("Saved Figures", paste(figures, collapse = ", "), "to", normalizePath(output_dir), "\n")
