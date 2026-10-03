#!/usr/bin/env Rscript
# Summarize the study-only experiment using its actual sample/taxon dimensions.
f <- grep('^--file=',commandArgs(FALSE),value=TRUE)[[1]]
f <- gsub('~+~',' ',sub('^--file=','',f),fixed=TRUE)
root <- dirname(dirname(normalizePath(f)))
run <- Sys.getenv('TWO_SAMPLE_RUN_ROOT',file.path(root,'output','ibdmdb_study'))
inputs <- file.path(run,'inputs')
read_matrix <- function(path) as.matrix(read.csv(path,header=FALSE))
prep <- dget(file.path(run,'preprocessing_settings.R'))
intervals <- diagnostics <- list()
for(split in c('train','test')) for(key in c('d','dt','icfm','mbgan')) {
 real <- read_matrix(file.path(inputs,paste0('sample_',split,'.csv')))
 generated <- read_matrix(file.path(inputs,paste0('sample_',key,'.csv')))
 z <- read_matrix(file.path(run,'revision',split,paste0('log_w_',key,'.csv')))
 stopifnot(ncol(real)==prep$n_taxa,ncol(generated)==prep$n_taxa,
           nrow(z)==nrow(real)+nrow(generated),ncol(z)==10L,all(is.finite(z)),
           all(z[,4:9,drop=FALSE]<=z[,5:10,drop=FALSE]))
 for(group in c('Real','Generated')) {
  idx <- if(group=='Real') seq_len(nrow(real)) else seq.int(nrow(real)+1L,nrow(z))
  intervals[[length(intervals)+1L]] <- data.frame(split=split,generator=key,source=group,n=length(idx),
   contain_zero=sum(z[idx,4]<=0 & z[idx,10]>=0),
   mean_abs_log_ratio=mean(abs(z[idx,3])),mean_interval_width=mean(z[idx,10]-z[idx,4]))
 }
 for(estimator in c('gradient','hellinger')) {
  saved <- readRDS(file.path(run,'fits',split,key,paste0(estimator,'.rds')))
  curve <- colMeans(saved$fit$loss_CV_store)
  stopifnot(all(is.finite(curve)))
  diagnostics[[length(diagnostics)+1L]] <- data.frame(split=split,generator=key,estimator=estimator,
   selected_trees=which.min(curve),minimum_cv_loss=min(curve),selected_at_search_limit=which.min(curve)==length(curve))
 }
}
ci <- do.call(rbind,intervals);cv <- do.call(rbind,diagnostics)
write.csv(ci,file.path(run,'interval_summary.csv'),row.names=FALSE)
write.csv(cv,file.path(run,'boosting_cv_summary.csv'),row.names=FALSE)
null <- read.csv(file.path(run,'null','test_pointwise_credible_intervals.csv'))
stopifnot(nrow(null)==prep$n_test,identical(null$test_row,seq_len(prep$n_test)),
          all(is.finite(as.matrix(null[,c('posterior_mean','ci_lower_025','ci_upper_975')]))),
          all(null$ci_lower_025<=null$ci_upper_975))
coverage <- sum(null$ci_lower_025<=0 & null$ci_upper_975>=0)
figures <- c('6','7','S8','S9','S10','S11','S12')
stopifnot(all(file.exists(file.path(run,'figures',paste0('figure',figures,'_ggplot2.pdf')))))
writeLines(c(
 '# IBDMDB-only experiment','',
 sprintf('The full %s resource was filtered within this study before splitting: %d original observations and %d original taxa; %d retained observations and %d retained taxa.',
  prep$resource,prep$original_dimensions[2],prep$original_dimensions[1],prep$n_selected,prep$n_taxa),'',
 sprintf('The 80/20 observation split contains %d training and %d test observations (seed %d). Repeated observations from the same participant can occur in both splits, matching the original observation-level design.',prep$n_train,prep$n_test,prep$split_seed),'',
 'The original filters are preserved, including retaining observations with more than 10% zero taxa. See preprocessing_settings.R and the two filter audit tables. Compositions are normalized over retained taxa and rounded to float32.', '',
 'All four generators were fitted afresh on sample_train.csv. The neural checkpoints are newly trained for this experiment; test observations are excluded from generator fitting. Checkpoint and data hashes, training budgets, commands and software versions are recorded in generator completion manifests, training metadata, logs and experiment_provenance.json.', '',
 paste0('BATTS source commit: `',readLines(file.path(run,'BATTS_commit.txt')),'`.'),'',
 'Each train/test versus generator comparison is fitted separately with Bayesian trees, gradient boosting and Hellinger boosting, following the existing case-study workflow. The test comparisons are density-ratio fits on real test observations versus generated observations; the generators remain trained only on the real training split.', '',
 'Comparisons use seed 1, 200 Bayesian trees, 2,000 burn-in and 1,000 retained iterations, thinning 1, lambda 5 and margin 0.1. Boosting uses five-fold CV, at most 1,000 trees, depth 4, learning rate 0.01 and 32 bins.', '',
 '## Descriptive pointwise intervals','',
 'These counts describe the real-versus-generated comparisons. The separate real-versus-real null experiment supplies the null coverage check.','',
 '| Split | Generator | Source | n | Intervals containing zero | Mean absolute log ratio | Mean interval width |',
 '| --- | --- | --- | ---: | ---: | ---: | ---: |',
 apply(ci,1,function(z)paste0('| ',paste(z,collapse=' | '),' |')),'',
 sprintf('The held-out null experiment has %d/%d intervals containing zero (%.1f%%). It uses seed 2026, balanced random training labels, 200 trees, 500 burn-in and 500 retained iterations, and evaluates all held-out real test observations.',coverage,prep$n_test,100*coverage/prep$n_test),'',
 sprintf('%d of %d boosting fits selected the maximum number of trees in the supplied search range.',sum(cv$selected_at_search_limit),nrow(cv)),'',
 '## Figures','',
 vapply(figures,function(k)sprintf('- [Figure %s](figures/figure%s_ggplot2.pdf)',k,k),character(1)),'',
 '## Provenance','',
 '- preprocessing_settings.R, sample_filter_audit.csv, taxon_filter_audit.csv: full-study filtering.',
 '- inputs/samples.csv and inputs/taxa.txt: observation membership and ordered taxa.',
 '- tree/: pruned phylogeny and generator tree transforms.',
 '- generators/: new checkpoints and training metadata; generator_*_complete.json: hashes.',
 '- revision/: ten-column estimator/quantile tables; fits/: cached fitted objects.',
 '- interval_summary.csv and boosting_cv_summary.csv: comparison summaries.',
 '- null/: null posterior draws, intervals and coverage.',
 '- logs/, sessionInfo.txt, BATTS_commit.txt and experiment_provenance.json: commands and settings.'
 ),file.path(run,'REPORT.md'))
print(ci,row.names=FALSE)
