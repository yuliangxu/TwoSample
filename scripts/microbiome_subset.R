library(curatedMetagenomicData)
library(dplyr)
library(DT)

library(ape)
library(phangorn)

###########################################################
#### 1. read dataset

#### think about 3 versions.

## (1) original
## (2) logistic tree transformation
## (3) clr

abundance_data <-
  curatedMetagenomicData(".relative_abundance",
                         dryrun = F, counts = F,
                         rownames = "long") |> mergeData()

samp_meta <- sampleMetadata |> select(c("study_name", "sample_id"))
samp_meta_v2 <- sampleMetadata |> select(where(~ !any(is.na(.x))))
samp_meta_v2 <- samp_meta_v2[, 1:7]

phylo_all <- abundance_data@rowTree$phylo
relative_abundance <-
  abundance_data@assays@data@listData$relative_abundance

###########################################################
#### 2. data property

dim(relative_abundance)


#####################################################
#### 3. subset taxa

col_idx1 <- (colSums(relative_abundance)>= 99)
row_idx1 <- rowMeans(relative_abundance[, col_idx1])  >= 0.1
sub_ra1 <- relative_abundance[row_idx1, ]

col_idx2 <- (colSums(sub_ra1 == 0) > dim(sub_ra1)[1]*0.1) & (colMeans(sub_ra1) >= mean(colMeans(sub_ra1)))
# col_idx2 <- (colMeans(sub_ra1) >= mean(colMeans(sub_ra1)))
col_sel_idx <- col_idx1 & col_idx2
# col_sel_idx <- col_idx1

ra_sub <- sub_ra1[, col_sel_idx]
dim(ra_sub)


min(colSums(ra_sub == 0))



samp_meta <- samp_meta[col_sel_idx, ]
samp_meta_v2 <- samp_meta_v2[col_sel_idx, ]
dim(samp_meta)
dim(samp_meta_v2)

species_of_interest <- rownames(ra_sub)
subtree <- keep.tip(phylo_all, species_of_interest)

