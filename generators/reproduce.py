"""Fit parametric generators and replay selected neural checkpoints.

This command regenerates the selected four synthetic matrices. Neural training
is intentionally a separate operation: the recovered checkpoints' original
training histories are unknown. See generators/README.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SELECTED = REPO/'data/regenerated_20261003'
SEEDS = {'d':1,'dt':1,'icfm':2,'mbgan':256}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'output/generators/reproduced')
    parser.add_argument('--models',nargs='+',choices=list(SEEDS),default=list(SEEDS))
    parser.add_argument('--tensorflow-python',default=sys.executable,help='Python interpreter from the MBGAN environment.')
    parser.add_argument('--include-real',action='store_true',help='Copy the unchanged public train/test CSVs into the output directory.')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be new or empty; existing outputs are never overwritten.')
    if len(set(args.models))!=len(args.models):
        parser.error('Do not repeat model names.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoints').mkdir()
    (out/'logs').mkdir()
    # Resolve relative interpreter paths before subprocess cwd changes.
    tf_python=args.tensorflow_python
    if '/' in tf_python:
        # Preserve a virtual environment's python symlink: resolving it to the
        # system executable loses the environment and its TensorFlow packages.
        tf_python=os.path.abspath(tf_python)
    report={'status':'running','models':{},'real_training_sha256':sha256(REPO/'data/sample_train.csv'),
            'neural_mode':'Recovered checkpoint replay. This does not retrain either historical neural model.'}
    for key in args.models:
        csv_path=out/f'sample_{key}.csv'
        if key in ['d','dt']:
            module='dirichlet' if key=='d' else 'dirichlet_tree'
            command=[sys.executable,'-m',f'generators.{module}','fit-sample',
                     '--train',str(REPO/'data/sample_train.csv'),
                     '--checkpoint',str(out/'checkpoints'/f'{key}.npz'),
                     '--output',str(csv_path),'--n-samples','1000','--seed','1',
                     '--report',str(out/f'{key}_fit.json')]
        elif key=='icfm':
            command=[sys.executable,'-m','generators.icfm','sample','--output',str(csv_path),
                     '--n','1000','--seed','2','--device','cpu','--threads','2']
        else:
            command=[tf_python,'-m','generators.mbgan','sample','--output',str(csv_path),
                     '--n','1000','--seed','256','--batch-size','32','--threads','2','--device','cpu']
        print('Generating',key,flush=True)
        with (out/'logs'/f'{key}.log').open('w') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        sample=np.loadtxt(csv_path,delimiter=',')
        reference=np.loadtxt(SELECTED/f'sample_{key}.csv',delimiter=',')
        if sample.shape!=(1000,123) or not np.isfinite(sample).all() or np.any(sample<0):
            raise ValueError(f'Invalid {key} synthetic matrix')
        if np.max(np.abs(sample.sum(1)-1))>1e-5:
            raise ValueError(f'Invalid {key} composition row sums')
        difference=np.abs(sample-reference)
        # Precision tolerances describe replay of the selected arrays, not fit
        # to the historical paper samples. Exact CSV agreement is recorded too.
        tolerance={'d':1e-9,'dt':1e-8,'icfm':1e-6,'mbgan':1e-6}[key]
        passed=bool(difference.max()<=tolerance)
        report['models'][key]={'seed':SEEDS[key],'shape':list(sample.shape),
            'sha256':sha256(csv_path),'selected_csv_sha256':sha256(SELECTED/f'sample_{key}.csv'),
            'exact_numeric_match':bool(np.array_equal(sample,reference)),
            'max_absolute_difference':float(difference.max()),'absolute_tolerance':tolerance,'passed':passed}
        (out/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
        if not passed:
            raise RuntimeError(f'{key} exceeds the selected-sample replay tolerance; inspect {out}/logs and package versions.')
    if args.include_real:
        for split in ['train','test']:
            shutil.copyfile(REPO/'data'/f'sample_{split}.csv',out/f'sample_{split}.csv')
    report['status']='complete'
    (out/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'COMPLETE').write_text('All requested generators passed selected-sample replay checks.\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
