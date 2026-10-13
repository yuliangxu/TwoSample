# Cluster execution

Install the pinned environments and BATTS commit as described in the root README.
Edit the account, partition and module loads for your cluster, then submit
`sbatch hpc/case_study.slurm` from the repository root. The default uses bundled
verified inputs. To regenerate them first, run the documented generator command
and set `TWO_SAMPLE_DATA_DIR` to its output directory.

Set `TWO_SAMPLE_RUN_ROOT` to a new durable scratch directory and `REPORT_PYTHON`
to the report environment interpreter. Eight workers use up to eight CPU cores;
no GPU is needed for the fixed CPU replay or BATTS analysis. All generated files
stay in the specified output directory. The final completion marker requires
both numerical validation and a successful rendered report.
