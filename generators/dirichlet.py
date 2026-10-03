"""Portable recovered Newton Dirichlet fit and legacy NumPy sampler.

The selected seed-1 output numerically reproduces the paper's Dirichlet sample
within about 8.71e-11; it is not bitwise equal to that historical sample.
Only --train is read for fitting, and no reference synthetic data are needed.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.special import psi,polygamma

FORMAT = "twosample-dirichlet-v1"
DEFAULT_TRAIN = Path(__file__).resolve().parents[1]/"data"/"sample_train.csv"


def _moments_concentration(data,eps=1e-8):
    mean = np.mean(data,axis=0)
    variance = np.var(data,axis=0,ddof=1)
    ratios = []
    for j in range(len(mean)):
        if variance[j]>eps and mean[j]>eps and mean[j]<1-eps:
            ratio = mean[j]*(1-mean[j])/variance[j]-1
            if ratio>0:
                ratios.append(ratio)
    value = np.median(ratios) if ratios else 10.0
    return np.clip(value,1.0,1000.0)


def fit(train,tol=1e-7,max_iter=1000):
    """Recovered damped Newton method, with additional final validity checks."""
    train = np.asarray(train,dtype=np.float64)
    if train.ndim!=2 or min(train.shape)<2 or not np.isfinite(train).all() or np.any(train<0):
        raise ValueError("Expected a finite nonnegative sample-by-taxon training array.")
    if not np.allclose(train.sum(axis=1),1,rtol=0,atol=1e-6):
        raise ValueError("Training rows must already be relative abundances summing to one.")
    if tol<=0 or max_iter<=0:
        raise ValueError("tol and max_iter must be positive.")
    eps = 1e-10
    # Preserve the recovered fit: clipping is not followed by renormalization.
    data = np.clip(train,eps,1.0)
    mean = np.mean(data,axis=0)
    alpha = mean*_moments_concentration(data)
    mean_log = np.mean(np.log(np.maximum(data,eps)),axis=0)
    converged_step = False
    for iteration in range(max_iter):
        alpha0 = np.sum(alpha)
        gradient = psi(alpha0)-psi(alpha)+mean_log
        q = -polygamma(1,alpha)
        z = polygamma(1,alpha0)
        q_safe = np.where(np.abs(q)<eps,-eps,q)
        z_safe = z if z>eps else eps
        denominator = 1/z_safe+np.sum(1/q_safe)
        if np.abs(denominator)<eps:
            denominator = eps
        b = np.sum(gradient/q_safe)/denominator
        update = (gradient-b)/q_safe
        alpha_new = alpha-update
        if np.any(alpha_new<=eps):
            step = 1.0
            while np.any(alpha-step*update<=eps) and step>1e-8:
                step *= 0.5
            alpha_new = np.maximum(alpha-step*update,eps)
        if np.max(np.abs(alpha_new-alpha))<tol:
            alpha = alpha_new
            converged_step = True
            break
        alpha = alpha_new
    residual = float(np.max(np.abs(psi(alpha.sum())-psi(alpha)+mean_log)))
    if not np.isfinite(alpha).all() or np.any(alpha<=0) or not converged_step or residual>=1e-8:
        raise RuntimeError(f"Unverified Dirichlet fit: step convergence={converged_step}, score residual={residual}")
    metadata = {"format":FORMAT,"training_shape":list(train.shape),
                "model":"Dirichlet","preprocessing":"clip entries to[1e-10,1]; no subsequent renormalization, matching recovered implementation",
                "solver":"Recovered Newton update with positivity step-halving; moments concentration initialization; verify final score residual<1e-8",
                "tol":tol,"max_iter":max_iter,"iterations":iteration+1,
                "max_equation_residual":residual,"all_parameters_verified":True,
                "parameter_min":float(alpha.min()),"parameter_max":float(alpha.max()),
                "concentration":float(alpha.sum()),
                "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}}
    return alpha,metadata


def sample(alpha,n_samples=1000,seed=1):
    alpha = np.asarray(alpha,dtype=np.float64)
    if alpha.ndim!=1 or len(alpha)<2 or not np.isfinite(alpha).all() or np.any(alpha<=0):
        raise ValueError("Dirichlet parameters must be a finite positive vector.")
    if n_samples<=0:
        raise ValueError("n_samples must be positive.")
    result = np.random.RandomState(seed).dirichlet(alpha,size=n_samples)
    if not np.isfinite(result).all() or np.any(result<0) or not np.allclose(result.sum(axis=1),1,atol=1e-12,rtol=0):
        raise RuntimeError("Generated observations failed simplex checks.")
    return result


def save_checkpoint(path,alpha,metadata):
    path = Path(path)
    if path.suffix!=".npz":
        raise ValueError("Checkpoint path must end in .npz.")
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,parameters=alpha,metadata=np.array(json.dumps(metadata,sort_keys=True)))


def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as state:
        alpha = state["parameters"]
        metadata = json.loads(str(state["metadata"].item()))
    if metadata.get("format")!=FORMAT:
        raise ValueError("Unsupported Dirichlet checkpoint format.")
    return alpha,metadata


def _save_samples(path,values):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix==".npy":
        np.save(path,values)
    elif path.suffix==".csv":
        np.savetxt(path,values,delimiter=",",fmt="%.18e")
    else:
        raise ValueError("Sample output must end in .npy or .csv.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    for command in ["fit","sample","fit-sample"]:
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint",type=Path,required=True)
        if command!="sample":
            sub.add_argument("--train",type=Path,default=DEFAULT_TRAIN)
        if command!="fit":
            sub.add_argument("--output",type=Path,required=True)
            sub.add_argument("--n-samples",type=int,default=1000)
            sub.add_argument("--seed",type=int,default=1)
        sub.add_argument("--report",type=Path)
    args = parser.parse_args()
    if args.command!="sample":
        train = np.load(args.train) if args.train.suffix==".npy" else np.loadtxt(args.train,delimiter=",")
        alpha,metadata = fit(train)
        metadata["training_file_name"] = args.train.name
        metadata["training_file_sha256"] = hashlib.sha256(args.train.read_bytes()).hexdigest()
        save_checkpoint(args.checkpoint,alpha,metadata)
    else:
        alpha,metadata = load_checkpoint(args.checkpoint)
    report = dict(metadata)
    if args.command!="fit":
        values = sample(alpha,args.n_samples,args.seed)
        _save_samples(args.output,values)
        report["sampling"] = {"seed":args.seed,"shape":list(values.shape),
                              "rng":"NumPy legacy RandomState MT19937, rowwise dirichlet draws",
                              "output_file_name":args.output.name,
                              "output_file_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
