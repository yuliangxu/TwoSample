file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
# Rscript may encode spaces as ~+~ in its --file argument.
script_path <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
project_root <- normalizePath(file.path(script_dir, ".."))
figures <- "S11"
source(file.path(script_dir, "lib", "quantile_figures.R"))
