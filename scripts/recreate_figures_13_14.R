file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
project_root <- normalizePath(file.path(script_dir, ".."))
suppressPackageStartupMessages({ library(ggplot2); library(scales) })

ratio_root <- file.path(project_root, "data", "revision")
pcoa_path <- file.path(project_root, "output", "figures_6_7", "pcoa_coordinates.rds")
output_dir <- file.path(project_root, "output", "figures_13_14")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(pcoa_path)) stop("Run recreate_figures_6_7.R first to create PCoA coordinates.")

keys <- c("d", "dt", "icfm", "mbgan")
method_labels <- c(d = "Dirichlet", dt = "Dirichlet Tree", icfm = "ICFM", mbgan = "MB-GAN")
real_n <- sapply(c("train", "test"), function(split)
  nrow(read.csv(file.path(project_root, "data", paste0("sample_", split, ".csv")), header = FALSE)))
quantity_labels <- c(mean = "Posterior mean", lower = "Lower 2.5% quantile", upper = "Upper 97.5% quantile")
ratio_colors <- c("darkblue", "blue", "deepskyblue", "cyan", "white", "#FFFF99", "yellow", "orange", "red", "darkred")

ordination <- readRDS(pcoa_path)
ordination$split <- tolower(sub(" vs\\. Generated$", "", as.character(ordination$split)))

load_panel_data <- function(split, key) {
  ratio <- as.matrix(read.csv(file.path(ratio_root, split, paste0("log_w_", key, ".csv")), header = FALSE))
  coords <- ordination[ordination$split == split & ordination$key == key, ]
  if (ncol(ratio) < 10L || nrow(coords) != nrow(ratio)) stop("PCoA/ratio row mismatch for ", split, "/", key)
  values <- list(mean = ratio[, 3], lower = ratio[, 4], upper = ratio[, 10])
  index <- rep(seq_len(nrow(ratio)), times = length(values))
  quantity <- rep(names(values), each = nrow(ratio))
  data.frame(
    PC1 = coords$PC1[index], PC2 = coords$PC2[index],
    source = ifelse(index <= real_n[[split]], "Real", "Generated"),
    method = method_labels[[key]], key = key, split = split,
    quantity = quantity_labels[quantity],
    log_ratio = unlist(values, use.names = FALSE)
  )
}

make_figure <- function(split, figure_number, color_limit) {
  plot_data <- do.call(rbind, lapply(keys, function(key) load_panel_data(split, key)))
  plot_data$method <- factor(plot_data$method, levels = unname(method_labels))
  plot_data$quantity <- factor(plot_data$quantity, levels = unname(quantity_labels))

  color_breaks <- if (color_limit == 30) c(-30, -10, 0, 10, 30) else c(-12, -4, 0, 4, 12)
  p <- ggplot(plot_data, aes(PC1, PC2, color = log_ratio)) +
    geom_point(size = 0.72, alpha = 0.7) +
    facet_grid(quantity ~ method, scales = "free") +
    scale_color_gradientn(
      colors = ratio_colors, limits = c(-color_limit, color_limit),
      values = rescale(c(-color_limit, 0, color_limit)),
      oob = squish, breaks = color_breaks,
      transform = pseudo_log_trans(sigma = color_limit * 0.01),
      guide = guide_colorbar(direction = "vertical", barheight = grid::unit(60, "mm"),
                             barwidth = grid::unit(6, "mm"), title.position = "top")
    ) +
    labs(x = "PCoA 1", y = "PCoA 2", color = "Log density ratio") +
    theme_bw(base_size = 15) +
    theme(
      panel.grid = element_blank(),
      strip.background = element_rect(fill = "white"),
      strip.text.x = element_text(face = "bold", size = 13),
      strip.text.y = element_text(face = "bold", size = 12),
      legend.position = "right", legend.text = element_text(size = 12),
      legend.title = element_text(size = 13)
    )

  output_path <- file.path(output_dir, paste0("figure", figure_number, "_ggplot2.pdf"))
  ggsave(output_path, p, width = 14, height = 9)
  write.csv(plot_data, file.path(output_dir, paste0("figure", figure_number, "_plot_data.csv")), row.names = FALSE)
  cat("Saved Figure", figure_number, "to", normalizePath(output_path), "\n")
}

make_figure("train", 13, 30)
make_figure("test", 14, 12)
