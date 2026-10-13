#!/usr/bin/env Rscript
options(repos = c(CRAN = 'https://cloud.r-project.org'), timeout = 1200)
if(!requireNamespace('BiocManager',quietly=TRUE)) install.packages('BiocManager')
if(!requireNamespace('remotes',quietly=TRUE)) install.packages('remotes')
# Dated resources and numerical checksums bind the reconstruction even if a
# newer compatible Bioconductor installation is used.
BiocManager::install('curatedMetagenomicData', ask=FALSE, update=FALSE)
message('Validated data package version: 3.20.0. Reconstruction must pass saved numerical checksums.')
