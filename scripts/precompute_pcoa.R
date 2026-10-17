# Fill the existing figure caches using exactly the repository's algorithms.
# Outputs are atomically renamed; running serial plotting remains valid.
suppressPackageStartupMessages(library(vegan))
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file)))
source(file.path(root, "scripts", "lib", "case_study_config.R"))
args <- commandArgs(TRUE)
workers <- if (length(args)) as.integer(args[[1]]) else 1L
stopifnot(length(workers) == 1L, !is.na(workers), workers >= 1L)
data_dir <- runtime_path("TWO_SAMPLE_DATA_DIR")
figure_dir <- runtime_path("TWO_SAMPLE_FIGURE_DIR", file.path(runtime_path("TWO_SAMPLE_RUN_ROOT"), "figures"))
cache_dir <- file.path(figure_dir, "cache")
dir.create(cache_dir,recursive=TRUE,showWarnings=FALSE)
grid <- expand.grid(key=case_study$keys,split=c('train','test'),algorithm=c('corrected','classical'),stringsAsFactors=FALSE)
one <- function(i) {
 g <- grid[i,];add <- g$algorithm=='corrected'
 inputs <- file.path(data_dir,paste0('sample_',c(g$split,g$key),'.csv'))
 fingerprint <- list(inputs=unname(tools::md5sum(inputs)),algorithm=if(add)'bray_cmdscale_add_true_v1' else 'bray_cmdscale_add_false_v1')
 path <- file.path(cache_dir,paste0(if(add)'' else 'classical_',g$split,'_',g$key,'.rds'))
 cached <- if(file.exists(path))tryCatch(readRDS(path),error=function(e)NULL) else NULL
 if(!is.null(cached)&&identical(cached$fingerprint,fingerprint))return(path)
 message(Sys.time(),' precompute ',g$algorithm,' ',g$split,'/',g$key)
 matrices <- lapply(inputs,function(p)as.matrix(read.csv(p,header=FALSE)))
 fit <- cmdscale(vegdist(do.call(rbind,matrices),method='bray'),k=2,eig=TRUE,add=add)
 if(add) {
  positive <- fit$eig[fit$eig>0]
  value <- list(fingerprint=fingerprint,ordination=list(points=fit$points[,1:2,drop=FALSE],pct=100*fit$eig[1:2]/sum(positive)))
 } else value <- list(fingerprint=fingerprint,points=fit$points)
 temp <- paste0(path,'.parallel-',Sys.getpid())
 saveRDS(value,temp)
 stopifnot(file.rename(temp,path))
 message(Sys.time(),' cached ',g$algorithm,' ',g$split,'/',g$key)
 path
}
result <- if (.Platform$OS.type != "windows" && workers > 1L)
  parallel::mclapply(seq_len(nrow(grid)), one, mc.cores=min(workers,nrow(grid)), mc.preschedule=FALSE) else
  lapply(seq_len(nrow(grid)), one)
stopifnot(!any(vapply(result,inherits,logical(1),'try-error')))
cat('All PCoA caches prepared with unchanged repository algorithms.\n')
