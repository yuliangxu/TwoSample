#!/usr/bin/env Rscript
# Install experiment dependencies into an HPC-native R library.
if (getRversion() < '4.6.0' || getRversion() >= '4.7.0')
 stop('Use R 4.6.x for this pinned Bioconductor 3.23 environment.')
args <- commandArgs(TRUE)
get_arg <- function(name,default) {
 x <- args[startsWith(args,paste0('--',name,'='))]
 if(length(x)) sub(paste0('^--',name,'='),'',x[[1]]) else default
}
lib <- get_arg('lib-dir','output/ibdmdb_study/r_lib')
dir.create(lib,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(normalizePath(lib),.libPaths()))
options(repos=c(CRAN='https://cloud.r-project.org'),timeout=1200)
install.packages(c('BiocManager','remotes','ggplot2','vegan','patchwork','scales','ape'),lib=lib)
# The dated IBDMDB resource, taxonomic labels and tree were audited with 3.20.0.
# Validated with R 4.6 and Bioconductor 3.23; pin the audited source package.
BiocManager::install(version='3.23',lib=lib,ask=FALSE,update=FALSE)
options(repos=BiocManager::repositories(version='3.23'))
remotes::install_url('https://bioconductor.org/packages/3.23/data/experiment/src/contrib/curatedMetagenomicData_3.20.0.tar.gz',lib=lib,upgrade='never')
stopifnot(packageVersion('curatedMetagenomicData')=='3.20.0')
remotes::install_github('nawaya040/BATTS@77c217297910a5ba50b71313e8289d9024b669c9',lib=lib,upgrade='never')
stopifnot(packageDescription('BATTS')$RemoteSha=='77c217297910a5ba50b71313e8289d9024b669c9')
writeLines(capture.output(sessionInfo()),file.path(lib,'experiment_setup_sessionInfo.txt'))
