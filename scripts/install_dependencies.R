#!/usr/bin/env Rscript
options(repos = c(CRAN = 'https://cloud.r-project.org'))
packages <- c('remotes', 'ggplot2', 'vegan', 'patchwork', 'scales', 'jsonlite', 'digest')
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly=TRUE)]
if(length(missing)) install.packages(missing)
commit <- '147e511d211919afefb8decac6333e66d24919db'
remotes::install_github(paste0('nawaya040/BATTS@',commit), upgrade='never')
stopifnot(packageVersion('BATTS') == '0.3.0', packageDescription('BATTS')$RemoteSha == commit)
