#!/usr/bin/env python3
"""Fixed-model ICFM comparison against public real splits; no fitting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parents[1]

def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 123 and np.isfinite(x).all() and np.all(x >= 0)
    assert np.all(x.sum(1) > 0)
    return x / x.sum(1, keepdims=True)

def distance(a, b, metric):
    if metric == 'sqrt_euclidean':
        return cdist(np.sqrt(a), np.sqrt(b), 'euclidean')
    if metric == 'euclidean':
        return cdist(a, b, 'euclidean')
    if metric == 'bray_curtis':
        return cdist(a, b, 'cityblock') / 2
    raise ValueError(metric)

def energies(rr, rq, qq):
    m, n = rq.shape
    cross = float(rq.mean())
    within_r = float(rr.sum() / (m * (m-1)))
    within_q = float(qq.sum() / (n * (n-1)))
    return {'u': 2*cross-within_r-within_q,
            'v': float(2*cross-rr.mean()-qq.mean()),
            'mean_cross_distance': cross,
            'mean_within_real_distance_off_diagonal': within_r,
            'mean_within_synthetic_distance_off_diagonal': within_q}

def bootstrap_delta(rn, ro, nn, oo, weights):
    # Same real resample for both comparisons. Within-real energy cancels.
    wr, wn, wo = weights
    n, o = nn.shape[0], oo.shape[0]
    cross_n = np.einsum('ij,ij->i', wr @ rn, wn)
    cross_o = np.einsum('ij,ij->i', wr @ ro, wo)
    within_n = np.einsum('ij,ij->i', wn @ nn, wn) * n/(n-1)
    within_o = np.einsum('ij,ij->i', wo @ oo, wo) * o/(o-1)
    return 2*(cross_n-cross_o)-within_n+within_o

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=REPO/'output/generators/icfm_comparison')
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error('--bootstrap must be at least2')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        'original': REPO/'data/sample_icfm.csv',
        'regenerated': REPO/'data/regenerated_20261003/sample_icfm.csv',
        'train': REPO/'data/sample_train.csv',
        'test': REPO/'data/sample_test.csv',
    }
    arrays = {k: normalized(np.load(p, allow_pickle=False) if p.suffix == '.npy' else np.loadtxt(p, delimiter=',')) for k,p in paths.items()}
    assert {k: v.shape for k,v in arrays.items()} == {'original': (1000,123),'regenerated': (1000,123),'train': (1166,123),'test': (292,123)}
    metrics = ['sqrt_euclidean', 'euclidean', 'bray_curtis']
    result = {
        'scope': 'Compare frozen original/published and regenerated ICFM cohorts; no retraining, tuning, or BATTS fitting.',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'inputs': {k: {'path': str(p.relative_to(REPO)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'shape': list(arrays[k].shape)} for k,p in paths.items()},
        'regenerated_definition': 'Recovered MLP checkpoint; restored torchdyn1.0.6 dopri5; float32 CPU; atol=rtol=1e-4; seed2; 1000 samples.',
        'primary_metric': 'sqrt_euclidean',
        'normalization': 'Each abundance row divided by its row sum before any calculation.',
        'metric_definitions': {
            'sqrt_euclidean': 'Euclidean energy after elementwise sqrt of normalized compositions. Distance is sqrt(2) times Hellinger distance; joint-sensitive.',
            'euclidean': 'Euclidean energy in normalized abundance space; joint-sensitive.',
            'bray_curtis': 'Energy with0.5*L1 distance. Equals a sum of marginal energies and cannot detect dependence-only distribution differences.'},
        'energy_definition': 'U=2*sum cross/(m*n)-sum real pair distances/(m*(m-1))-sum synthetic pair distances/(n*(n-1)); V retains n^2 denominators. Lower is closer; finite-sample U may be negative.',
        'bootstrap': {'replicates': args.bootstrap, 'seed': 20261003, 'interval': '95% percentile interval for U(regenerated)-U(original)', 'resampling': 'Real rows sampled once per replicate and shared between comparisons; each synthetic cohort resampled independently with replacement.', 'limitations': 'Conditional on frozen generators and iid rows; reflects finite-cohort sampling, not training variability, historical data leakage, or provenance uncertainty. Descriptive intervals without multiple-comparison correction.'},
        'split_caveat': 'Public train/test roles are preserved. Historical checkpoint fitting membership is unknown, so independent test status relative to that checkpoint is unverified.',
        'comparisons': [],
    }
    t0=time.monotonic()
    self_dist = {(label,metric): distance(arrays[label],arrays[label],metric) for label in ['original','regenerated'] for metric in metrics}
    for value in self_dist.values():
        np.fill_diagonal(value,0.)
    for split_index,split in enumerate(['train','test']):
        real=arrays[split]
        matrices={metric: (distance(real,real,metric), distance(real,arrays['regenerated'],metric), distance(real,arrays['original'],metric)) for metric in metrics}
        for rr,_,_ in matrices.values():
            np.fill_diagonal(rr,0.)
        rng=np.random.default_rng(20261003+split_index)
        bootstrap={metric: [] for metric in metrics}
        for begin in range(0,args.bootstrap,64):
            batch=min(64,args.bootstrap-begin)
            weights=tuple(rng.multinomial(n,np.full(n,1/n),size=batch).astype(float)/n for n in [len(real),1000,1000])
            for metric,(_,rn,ro) in matrices.items():
                bootstrap[metric].append(bootstrap_delta(rn,ro,self_dist['regenerated',metric],self_dist['original',metric],weights))
        for metric,(rr,rn,ro) in matrices.items():
            new=energies(rr,rn,self_dist['regenerated',metric])
            old=energies(rr,ro,self_dist['original',metric])
            samples=np.concatenate(bootstrap[metric])
            np.save(args.output/f'bootstrap_delta_{split}_{metric}.npy',samples)
            row={'split': split,'metric': metric,'original': old,'regenerated': new,
                 'delta_u_regenerated_minus_original': new['u']-old['u'],
                 'percent_reduction_u': 100*(old['u']-new['u'])/old['u'],
                 'delta_u_bootstrap_95_percentile': np.quantile(samples,[.025,.975]).tolist()}
            result['comparisons'].append(row)
            print(json.dumps(row),flush=True)
        print(f'Completed {split} in {time.monotonic()-t0:.1f}s',flush=True)
    result['runtime_seconds']=time.monotonic()-t0
    (args.output/'joint_comparison.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    main()
