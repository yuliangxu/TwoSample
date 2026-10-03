#!/usr/bin/env Rscript
# Apply the original filtering rules to the full IBDMDB resource, before splitting.
f <- grep('^--file=', commandArgs(FALSE), value=TRUE)[[1]]
f <- gsub('~+~',' ',sub('^--file=','',f),fixed=TRUE)
root <- dirname(dirname(normalizePath(f)))
args <- commandArgs(TRUE)
arg <- function(key, default=NULL) {
  hits <- args[startsWith(args,paste0('--',key,'='))]
  if(length(hits)) sub(paste0('^--',key,'='),'',hits[[1]]) else default
}
.libPaths(c(Sys.getenv('TWO_SAMPLE_R_LIB',file.path(root,'output','ibdmdb_study','r_lib')),.libPaths()))
run <- arg('output',file.path(root,'output','ibdmdb_study'))
seed <- as.integer(arg('seed','20261003'))
stopifnot(!is.na(seed))
dir.create(run,recursive=TRUE,showWarnings=FALSE);run <- normalizePath(run)
if(run==normalizePath(file.path(root,'data'))) stop('Output cannot be the bundled data directory')
inputs <- file.path(run,'inputs');dir.create(inputs,showWarnings=FALSE)
cache <- arg('cache-dir')
if(!is.null(cache)) {dir.create(cache,recursive=TRUE,showWarnings=FALSE);Sys.setenv(EXPERIMENT_HUB_CACHE=normalizePath(cache))}
suppressPackageStartupMessages({library(curatedMetagenomicData);library(ape)})
if(packageVersion('curatedMetagenomicData')!='3.20.0')
 stop('Use curatedMetagenomicData 3.20.0; install with scripts/setup_ibdmdb.R.')
ExperimentHub::setExperimentHubOption('LOCAL','--offline' %in% args)
options(timeout=max(1200,getOption('timeout')))
resource <- '2021-10-14.HMP_2019_ibdmdb.relative_abundance'
loaded <- curatedMetagenomicData('^2021-10-14\\.HMP_2019_ibdmdb\\.relative_abundance$',dryrun=FALSE,counts=FALSE,rownames='long')
stopifnot(identical(names(loaded),resource));se <- loaded[[1]]
metadata <- as.data.frame(SummarizedExperiment::colData(se))
ra <- as.matrix(SummarizedExperiment::assay(se))
stopifnot(identical(rownames(metadata),colnames(ra)),all(metadata$study_name=='HMP_2019_ibdmdb'))
# Exact original operators/reference populations, restricted to IBDMDB first.
col_idx1 <- colSums(ra)>=99
stopifnot(any(col_idx1))
row_idx1 <- rowMeans(ra[,col_idx1,drop=FALSE])>=0.1
sub_ra1 <- ra[row_idx1,,drop=FALSE]
col_idx2 <- (colSums(sub_ra1==0)>nrow(sub_ra1)*0.1) &
            (colMeans(sub_ra1)>=mean(colMeans(sub_ra1)))
selected <- col_idx1 & col_idx2
filtered <- sub_ra1[,selected,drop=FALSE]
stopifnot(ncol(filtered)>1L,nrow(filtered)>1L)
x <- t(filtered);x <- x/rowSums(x)
x[] <- readBin(writeBin(as.double(x),raw(),size=4,endian='little'),'double',n=length(x),size=4,endian='little')
set.seed(seed);permutation <- sample.int(nrow(x))
n_train <- as.integer(round(nrow(x)*0.8))
train_idx <- permutation[seq_len(n_train)];test_idx <- permutation[-seq_len(n_train)]
write_matrix <- function(z,path) writeLines(apply(z,1,function(row)paste(sprintf('%.18e',row),collapse=',')),path)
write_matrix(x[train_idx,,drop=FALSE],file.path(inputs,'sample_train.csv'))
write_matrix(x[test_idx,,drop=FALSE],file.path(inputs,'sample_test.csv'))
writeLines(colnames(x),file.path(inputs,'taxa.txt'))
meta <- metadata[selected,,drop=FALSE]
manifest <- rbind(data.frame(split='train',csv_row=seq_along(train_idx),sample_id=rownames(x)[train_idx],study_name=meta$study_name[train_idx]),data.frame(split='test',csv_row=seq_along(test_idx),sample_id=rownames(x)[test_idx],study_name=meta$study_name[test_idx]))
stopifnot(!anyDuplicated(manifest$sample_id))
write.csv(manifest,file.path(inputs,'samples.csv'),row.names=FALSE)
write.csv(data.frame(sample_id=colnames(ra),total_abundance=colSums(ra),passes_total=col_idx1,retained_taxon_zero_count=colSums(sub_ra1==0),retained_taxon_mean=colMeans(sub_ra1),passes_second_filter=col_idx2,selected=selected),file.path(run,'sample_filter_audit.csv'),row.names=FALSE)
write.csv(data.frame(taxon=rownames(ra),mean_among_complete_samples=rowMeans(ra[,col_idx1,drop=FALSE]),selected=row_idx1),file.path(run,'taxon_filter_audit.csv'),row.names=FALSE)
# Prune the package phylogeny and deterministically resolve any multifurcations.
phylo <- TreeSummarizedExperiment::rowTree(se)
if(is.list(phylo)&&!inherits(phylo,'phylo')) phylo <- phylo[[1]]
subtree <- ape::keep.tip(phylo,colnames(x))
subtree <- ape::multi2di(subtree,random=FALSE)
p <- length(subtree$tip.label)
stopifnot(p==ncol(x),subtree$Nnode==p-1L,setequal(subtree$tip.label,colnames(x)))
children <- split(subtree$edge[,2],subtree$edge[,1])
root_node <- setdiff(subtree$edge[,1],subtree$edge[,2])
stopifnot(length(root_node)==1L)
# Internal nodes must be numbered in preorder for the existing TreeTransform.
records <- vector('list',p-1L);counter <- 0L
visit <- function(node) {
 if(node<=p) return(paste0('Node_',node))
 counter <<- counter+1L;index <- counter;name <- paste0('Node_',p+index)
 pair <- children[[as.character(node)]];stopifnot(length(pair)==2L)
 left <- visit(pair[[1]]);right <- visit(pair[[2]])
 records[[index]] <<- c(name,left,right)
 name
}
visit(root_node)
stopifnot(counter==p-1L)
tree_dir <- file.path(run,'tree');dir.create(tree_dir,showWarnings=FALSE)
record_matrix <- do.call(rbind,records)
write.csv(data.frame(Child_1=record_matrix[,2],Child_2=record_matrix[,3],row.names=record_matrix[,1]),file.path(tree_dir,'childNode.csv'))
write.csv(data.frame(x=subtree$tip.label),file.path(tree_dir,'treeTaxa.csv'))
write.csv(data.frame(x=colnames(x)),file.path(tree_dir,'outTaxa.csv'))
saveRDS(subtree,file.path(tree_dir,'study_tree.rds'))
ape::write.tree(subtree,file.path(tree_dir,'study_tree.nwk'))
settings <- list(study='HMP_2019_ibdmdb',resource=resource,package_version=as.character(packageVersion('curatedMetagenomicData')),original_dimensions=dim(ra),n_complete=sum(col_idx1),n_selected=nrow(x),n_taxa=ncol(x),n_train=n_train,n_test=length(test_idx),split_seed=seed,split_unit='observation',rules=c('total abundance >= 99','mean taxon abundance among complete samples >= 0.1','zero taxa count > 10% of retained taxa','retained-taxon sample mean >= study-wide mean across all original IBDMDB observations'),normalization='over retained taxa, then IEEE-754 float32',processing_population='full IBDMDB study only, before train/test split',binary_tree_resolution='ape::multi2di(random=FALSE); internal nodes numbered in preorder')
dput(settings,file.path(run,'preprocessing_settings.R'))
saveRDS(list(abundance=filtered,metadata=meta),file.path(run,'filtered_study.rds'))
writeLines(capture.output(sessionInfo()),file.path(run,'preprocessing_sessionInfo.txt'))
files <- c(list.files(inputs,full.names=TRUE),list.files(tree_dir,full.names=TRUE))
write.csv(data.frame(file=sub(paste0(run,'/'),'',files,fixed=TRUE),md5=unname(tools::md5sum(files))),file.path(run,'preprocessing_checksums.csv'),row.names=FALSE)
print(settings)
