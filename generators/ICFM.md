# Fixed ICFM checkpoint replay

The final ICFM cohort uses `checkpoints/icfm_recovered.pt`, whose SHA-256 is
`6f8d3b318687c910b0f549c922aff4d1f346a3773f7661505b8aa8d4af4118be`.
The time-dependent MLP has 122 log-tree coordinates plus one time input,
three width-64 SELU hidden layers, and 122 output velocities.

From the repository root, using the pinned generator environment:

```sh
python -m generators.icfm sample \
  --checkpoint generators/checkpoints/icfm_recovered.pt \
  --n 1000 --seed 2 --device cpu --threads 2 \
  --atol 1e-4 --rtol 1e-4 \
  --output output/icfm/sample_icfm.csv
```

The sampler draws CPU float32 standard normals, integrates from time 0 to 1
with the restored torchdyn 1.0.6 adaptive Dormand-Prince path, and reconstructs
123 taxon proportions. The three CSVs under `assets/tree/` fix taxon order and
tree structure. The inverse sigmoid is computed in float32; complements and
tree products are computed in float64. Preserving this arithmetic and drawing
all 1,000 initial states together is required for the selected numerical path.

The final CSV SHA-256 is
`24ca2b129d6073bedbe151eec23bf469fb1e79fe1c4a6ed53c2932a19436d58c`.
Writing `.npy` instead produces the reference NPY SHA-256
`c1043b018c7bb554577a1fc21cddc590c103c23bd66c8b70a351c12744e1418d`.
The output sidecar records the checkpoint, tree and source hashes, random seed,
solver tolerances, precision, software versions, device and composition checks.
Existing outputs are protected from overwriting.

The `vendor/` adapter selects the unmodified numerical definitions from the
preserved torchdyn 1.0.6 source. It does not require torchdyn, torchdiffeq,
PyTorch Lightning or event-callback dependencies to be installed. Source hashes,
modifications and the Apache-2.0 license are included. The original historical
notebook did not record its torchdyn version.

This is a fixed-checkpoint case study. The checkpoint's historical training
membership and complete fitting settings remain unverified. Restoring its
numerical sampling path does not establish recovery of the earlier published
synthetic cohort or original training run.
