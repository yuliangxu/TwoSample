"""Fixed binary-tree transformations shared by the microbiome generators.

The DT and archived ICFM preprocessing conventions intentionally differ.
Input compositions have samples in rows and taxa in outTaxa.csv order.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit

DEFAULT_TREE_DIR = Path(__file__).resolve().parent / "assets" / "tree"


def _rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.reader(handle))[1:]


class TreeTransform:
    """Validated fixed tree with explicit preprocessing and inverse precision."""

    def __init__(self, directory=DEFAULT_TREE_DIR):
        self.directory = Path(directory)
        children = _rows(self.directory / "childNode.csv")
        self.internal_nodes = [row[0] for row in children]
        self.taxa = [row[1] for row in _rows(self.directory / "treeTaxa.csv")]
        self.out_taxa = [row[1] for row in _rows(self.directory / "outTaxa.csv")]
        self.n_taxa = len(self.out_taxa)
        self.n_internal = len(children)
        if self.n_internal != self.n_taxa-1:
            raise ValueError("Expected a full binary tree with p-1 internal nodes.")
        if len(set(self.taxa)) != self.n_taxa or set(self.taxa) != set(self.out_taxa):
            raise ValueError("Tree and output taxon labels must be unique and equal.")
        expected = [f"Node_{j}" for j in range(self.n_taxa+1, 2*self.n_taxa)]
        if self.internal_nodes != expected:
            raise ValueError("Expected internal node labels in their recorded numeric order.")
        self.children = np.array([[int(v.removeprefix("Node_"))-1 for v in row[1:3]]
                                  for row in children], dtype=int)
        if self.children.shape != (self.n_internal,2):
            raise ValueError("Every internal node must have two children.")
        flat = self.children.ravel().tolist()
        root_candidates = set(range(self.n_taxa,2*self.n_taxa-1))-set(flat)
        if len(root_candidates)!=1 or len(set(flat))!=len(flat):
            raise ValueError("The tree must have one root and a unique parent per child.")
        self.root = root_candidates.pop()
        if set(flat)|{self.root} != set(range(2*self.n_taxa-1)):
            raise ValueError("Tree nodes do not form a connected labeled binary tree.")
        for j,pair in enumerate(self.children):
            if any(c>=self.n_taxa and c<=self.n_taxa+j for c in pair):
                raise ValueError("Historical reconstruction requires parent-before-child order.")
        self.input_order = np.array([self.out_taxa.index(name) for name in self.taxa])
        self.output_order = np.array([self.taxa.index(name) for name in self.out_taxa])

    @classmethod
    def from_directory(cls, directory):
        return cls(directory)

    def hashes(self):
        return {name: hashlib.sha256((self.directory/name).read_bytes()).hexdigest()
                for name in ["childNode.csv","treeTaxa.csv","outTaxa.csv"]}

    def _validate_compositions(self, x):
        x = np.asarray(x)
        if x.ndim!=2 or x.shape[1]!=self.n_taxa or not np.isfinite(x).all():
            raise ValueError(f"Expected finite sample-by-taxon array with {self.n_taxa} columns.")
        if np.any(x<0) or np.any(x.sum(axis=1)<=0):
            raise ValueError("Compositions must be nonnegative with positive row mass.")
        return x

    def child_masses(self, x):
        x = self._validate_compositions(x)
        mass = np.zeros((len(x),2*self.n_taxa-1),dtype=np.float64)
        mass[:,:self.n_taxa] = x[:,self.input_order]
        for j in range(self.n_internal-1,-1,-1):
            left,right = self.children[j]
            mass[:,self.n_taxa+j] = mass[:,left]+mass[:,right]
        return mass[:,self.children[:,0]],mass[:,self.children[:,1]]

    def forward_dt(self, x, epsilon=1e-7):
        """Chosen DT convention: float32 renormalization, exact-zero smoothing."""
        if not 0<epsilon<0.5:
            raise ValueError("epsilon must lie between zero and one half.")
        x = self._validate_compositions(x).astype(np.float32)
        x = x/x.sum(axis=1,keepdims=True)
        left,right = self.child_masses(x)
        left = np.where(left==0,epsilon,np.where(left==1,1-epsilon,left))
        right = np.where(right==0,epsilon,np.where(right==1,1-epsilon,right))
        return np.log(left/right)

    def forward_icfm(self, x, zero_mass=1e-8, clip_logratio=np.log(1e6)):
        """Archive-matching formula: zero masses replaced, log ratios clipped.

        This formula was inferred and verified against archived training LT;
        it is distinct from the newer DT transform. Normalize in float64.
        """
        if zero_mass<=0 or clip_logratio<=0:
            raise ValueError("zero_mass and clip_logratio must be positive.")
        x = self._validate_compositions(x).astype(np.float64)
        x = x/x.sum(axis=1,keepdims=True)
        left,right = self.child_masses(x)
        lt = np.log(np.where(left==0,zero_mass,left)/np.where(right==0,zero_mass,right))
        return np.clip(lt,-clip_logratio,clip_logratio)

    def forward(self, x):
        """Pure unsmoothed log ratios; zero-containing input may yield inf/NaN."""
        left,right = self.child_masses(x)
        with np.errstate(divide="ignore",invalid="ignore"):
            return np.log(left/right)

    def inverse(self, logratios, mode="float64"):
        """Reconstruct rows in public output order with explicit arithmetic mode.

        dt_float64 preserves the chosen DT's stable sigmoid expression.
        icfm_float32 computes sigmoid in float32, then complement and tree
        multiplication in float64, exactly as the historical notebook.
        float64 uses scipy.special.expit for new numerical work.
        """
        z = np.asarray(logratios)
        if z.ndim!=2 or z.shape[1]!=self.n_internal or not np.isfinite(z).all():
            raise ValueError(f"Expected finite sample-by-node array with {self.n_internal} columns.")
        with np.errstate(over="ignore",invalid="ignore"):
            if mode=="icfm_float32":
                z = z.astype(np.float32)
                theta = (1/(1+np.exp(-z))).astype(np.float64)
            elif mode=="dt_float64":
                z = z.astype(np.float64)
                theta = np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))
            elif mode=="float64":
                theta = expit(z.astype(np.float64))
            else:
                raise ValueError("mode must be dt_float64, icfm_float32, or float64")
        reconstructed = np.stack([theta,1-theta],axis=2)
        tips = np.zeros((len(z),self.n_taxa),dtype=np.float64)
        for j,pair in enumerate(self.children):
            for side,child in enumerate(pair):
                if child>=self.n_taxa:
                    reconstructed[:,child-self.n_taxa,:] *= reconstructed[:,j,side,None]
                else:
                    tips[:,child] = reconstructed[:,j,side]
        return tips[:,self.output_order]
