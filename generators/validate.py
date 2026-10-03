"""Validate regenerated matrices and compare them with public real data.

Run with ``python -m generators.validate``. This does not fit BATTS or a generator.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[1]
SELECTED = REPO / 'data' / 'regenerated_20261003'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path, n=None):
    x = np.loadtxt(path, delimiter=',')
    if x.ndim != 2 or x.shape[1] != 123 or (n is not None and len(x) != n):
        raise ValueError(f'Unexpected matrix shape: {path}: {x.shape}')
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x.sum(1) <= 0):
        raise ValueError(f'Invalid nonnegative compositions: {path}')
    if np.max(np.abs(x.sum(1)-1)) > 1e-5:
        raise ValueError(f'Rows do not sum to one: {path}')
    return x

def normalized(x):
    return x / x.sum(1, keepdims=True)

def shannon(x):
    return -(x*np.log(np.maximum(x, np.finfo(float).tiny))).sum(1)

def energy_u(x, y, geometry):
    x, y = normalized(x), normalized(y)
    metric, factor = 'euclidean', 1.
    if geometry == 'sqrt_euclidean':
        x, y = np.sqrt(x), np.sqrt(y)
    elif geometry == 'bray_curtis':
        metric, factor = 'cityblock', .5
    elif geometry != 'euclidean':
        raise ValueError(geometry)
    return float(factor*(2*cdist(x,y,metric).mean()
        -2*pdist(x,metric).sum()/(len(x)*(len(x)-1))
        -2*pdist(y,metric).sum()/(len(y)*(len(y)-1))))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=SELECTED)
    parser.add_argument('--output', type=Path, default=REPO/'output/generators/validation')
    parser.add_argument('--check-bundled-hashes', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((SELECTED/'manifest.json').read_text())
    reals={split:load(REPO/'data'/f'sample_{split}.csv',n) for split,n in [('train',1166),('test',292)]}
    files={}
    rows=[]
    for key in ['d','dt','icfm','mbgan']:
        path=args.data_dir/f'sample_{key}.csv'
        x=load(path,1000)
        ref=load(REPO/'data'/f'sample_{key}.csv',1000)
        hash_value=sha256(path)
        match=hash_value==manifest['models'][key]['sha256']
        if args.check_bundled_hashes and not match:
            raise ValueError(f'Bundled checksum mismatch: {key}')
        files[key]={'csv':path.name,'sha256':hash_value,'matches_selected_csv_checksum':match,
                    'shape':list(x.shape),'max_row_sum_error':float(np.abs(x.sum(1)-1).max()),
                    'max_absolute_difference_from_published':float(np.abs(x-ref).max())}
        for split,real in reals.items():
            rn=normalized(real)
            for version,synthetic in [('published',ref),('regenerated',x)]:
                sn=normalized(synthetic)
                row={'model':key,'real_split':split,'version':version,
                     'real_n':len(real),'synthetic_n':len(synthetic),
                     **{f'energy_U_{g}':energy_u(real,synthetic,g) for g in ['sqrt_euclidean','euclidean','bray_curtis']},
                     'mean_composition_L1':float(np.abs(rn.mean(0)-sn.mean(0)).sum()),
                     'taxon_prevalence_MAE_at_1e_minus4':float(np.abs((rn>1e-4).mean(0)-(sn>1e-4).mean(0)).mean()),
                     'shannon_mean_real':float(shannon(rn).mean()),'shannon_mean_synthetic':float(shannon(sn).mean()),
                     'shannon_wasserstein1':float(wasserstein_distance(shannon(rn),shannon(sn))),
                     'dominant_taxon_mean_real':float(rn.max(1).mean()),'dominant_taxon_mean_synthetic':float(sn.max(1).mean()),
                     'richness_mean_real':float((rn>1e-4).sum(1).mean()),'richness_mean_synthetic':float((sn>1e-4).sum(1).mean())}
                rows.append(row)
    result={'scope':'Fixed synthetic cohorts versus public real splits; no fitting or sample selection.',
            'energy_definition':'Off-diagonal unbiased energy U statistic; lower is closer. Finite-sample estimates may be negative.',
            'bray_curtis_caveat':'On normalized compositions this energy is a sum of coordinate marginal energies, not a general joint-distribution discriminator.',
            'split_caveat':'Parametric models use the public training split. Historical neural-checkpoint fitting membership is unverified.',
            'files':files,'comparisons':rows}
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    print(json.dumps({'files':files,'output':str(args.output)},indent=2))

if __name__ == '__main__':
    main()
