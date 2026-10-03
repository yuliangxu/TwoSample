"""Fit and sample the selected stabilized independent-Beta Dirichlet tree.

This is a repaired generator, not bitwise reproduction of the paper's original
DT realization. Training uses only the input passed to --train (public train by
default). Neither test observations nor synthetic reference arrays are read.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import betaln, polygamma, psi

from .tree import DEFAULT_TREE_DIR, TreeTransform

DEFAULT_TRAIN = Path(__file__).resolve().parents[1]/"data"/"sample_train.csv"
FORMAT = "twosample-dirichlet-tree-v1"


def fit_beta(x):
    """Positive log-parameter solve, accepted only at a verified interior root."""
    x = np.clip(np.asarray(x,dtype=np.float64),1e-10,1-1e-10)
    log_moments = np.array([np.log(x).mean(),np.log1p(-x).mean()])
    mean,variance = x.mean(),x.var()
    if not variance>0:
        raise ValueError("A constant split has no finite unconstrained Beta MLE.")
    concentration = max(mean*(1-mean)/variance-1,1e-2)
    initial = np.maximum([mean*concentration,(1-mean)*concentration],1e-4)

    def equations(log_params):
        params = np.exp(log_params)
        return psi(params)-psi(params.sum())-log_moments

    def jacobian(log_params):
        params = np.exp(log_params)
        shared = polygamma(1,params.sum())
        derivative = np.diag(polygamma(1,params))-shared
        return derivative*params[None,:]

    result = least_squares(equations,np.log(initial),jac=jacobian,bounds=(-30,30),
                           xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=1000)
    params = np.exp(result.x)
    residual = float(np.max(np.abs(equations(result.x))))
    interior = bool(np.all(np.abs(result.x)<29))
    valid = bool(result.success and np.isfinite(params).all() and (params>0).all()
                 and interior and residual<1e-8)
    if not valid:
        raise RuntimeError(f"Unverified Beta MLE: {result.message}; residual={residual}")
    return params,{"success":bool(result.success),"max_equation_residual":residual,
                   "interior":interior,"function_evaluations":int(result.nfev),
                   "parameters":params.tolist(),"log_likelihood_per_row":float(
                       -betaln(*params)+np.sum((params-1)*log_moments))}


def fit(train, tree=None):
    tree = tree or TreeTransform()
    train = np.asarray(train,dtype=np.float64)
    if len(train)<2:
        raise ValueError("At least two training observations are required.")
    lt = tree.forward_dt(train,epsilon=1e-7)
    theta = 1/(np.exp(-lt)+1)
    fitted = [fit_beta(theta[:,j]) for j in range(tree.n_internal)]
    params = np.stack([item[0] for item in fitted])
    nodes = [dict(node=j,**item[1]) for j,item in enumerate(fitted)]
    report = {"format":FORMAT,"training_shape":list(train.shape),
              "model":"Independent Beta splits on the fixed binary tree",
              "preprocessing":"float32 row normalization; child mass 0->1e-7,1->1-1e-7; log ratio; sigmoid; split clipping[1e-10,1-1e-10]",
              "solver":"least_squares in log shape parameters; bounds[-30,30]; moments start; analytic Jacobian; xtol=ftol=gtol=1e-12; max_nfev=1000; interior and residual<1e-8 required",
              "tree_hashes":tree.hashes(),"nodes":nodes,
              "max_equation_residual":max(node["max_equation_residual"] for node in nodes),
              "all_nodes_verified":True,"historical_reproduction":False,
              "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}}
    return params,report


def sample(params,n_samples=1000,seed=1,tree=None):
    tree = tree or TreeTransform()
    params = np.asarray(params,dtype=np.float64)
    if params.shape!=(tree.n_internal,2) or not np.isfinite(params).all() or np.any(params<=0):
        raise ValueError("Expected positive finite Beta parameters for every tree node.")
    if n_samples<=0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.RandomState(seed)
    theta = np.empty((n_samples,tree.n_internal))
    for j,(a,b) in enumerate(params):
        theta[:,j] = rng.beta(a,b,size=n_samples)
    theta = np.clip(theta,1e-10,1-1e-10)
    lt = np.log(theta/(1-theta))
    result = tree.inverse(lt,mode="dt_float64")
    if not np.isfinite(result).all() or np.any(result<0) or not np.allclose(result.sum(axis=1),1,atol=1e-12,rtol=0):
        raise RuntimeError("Generated observations failed simplex checks.")
    return result


def save_checkpoint(path,params,metadata):
    path = Path(path)
    if path.suffix!=".npz":
        raise ValueError("Checkpoint path must end in .npz.")
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,parameters=params,metadata=np.array(json.dumps(metadata,sort_keys=True)))


def load_checkpoint(path,tree=None):
    tree = tree or TreeTransform()
    with np.load(path,allow_pickle=False) as state:
        params = state["parameters"]
        metadata = json.loads(str(state["metadata"].item()))
    if metadata.get("format")!=FORMAT or metadata.get("tree_hashes")!=tree.hashes():
        raise ValueError("Checkpoint format or tree metadata does not match.")
    return params,metadata


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
        sub.add_argument("--tree",type=Path,default=DEFAULT_TREE_DIR)
        sub.add_argument("--checkpoint",type=Path,required=True)
        if command!="sample":
            sub.add_argument("--train",type=Path,default=DEFAULT_TRAIN)
        if command!="fit":
            sub.add_argument("--output",type=Path,required=True)
            sub.add_argument("--n-samples",type=int,default=1000)
            sub.add_argument("--seed",type=int,default=1)
        sub.add_argument("--report",type=Path)
    args = parser.parse_args()
    tree = TreeTransform.from_directory(args.tree)
    if args.command!="sample":
        train = np.load(args.train) if args.train.suffix==".npy" else np.loadtxt(args.train,delimiter=",")
        params,metadata = fit(train,tree)
        metadata["training_file_name"] = args.train.name
        metadata["training_file_sha256"] = hashlib.sha256(args.train.read_bytes()).hexdigest()
        save_checkpoint(args.checkpoint,params,metadata)
    else:
        params,metadata = load_checkpoint(args.checkpoint,tree)
    report = dict(metadata)
    if args.command!="fit":
        values = sample(params,args.n_samples,args.seed,tree)
        _save_samples(args.output,values)
        report["sampling"] = {"seed":args.seed,"shape":list(values.shape),"inverse":"dt_float64",
                              "rng":"NumPy legacy RandomState MT19937, nodewise beta draws",
                              "output_file_name":args.output.name,
                              "output_file_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key!="nodes"},indent=2))


if __name__=="__main__":
    main()
