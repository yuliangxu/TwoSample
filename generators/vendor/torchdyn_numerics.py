"""Load unmodified numerical definitions from preserved torchdyn 1.0.6 source.

The plain adaptive dopri5 path does not require attrs, event callbacks, torchcde,
or PyTorch Lightning. AST selection skips their unrelated imports/classes while
preserving every statement and arithmetic operation in the selected definitions.
"""
import ast
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union
from warnings import warn
import torch
from torch import Tensor, cat, nn

SOURCE = Path(__file__).resolve().parent / "torchdyn_1_0_6"
SELECTIONS = [
    ("torchdyn/numerics/solvers/templates.py", ["DiffEqSolver"]),
    ("torchdyn/numerics/solvers/_constants.py", ["construct_dopri5"]),
    ("torchdyn/numerics/solvers/ode.py", ["DormandPrince45", "Tsitouras45"]),
    ("torchdyn/numerics/utils.py", ["hairer_norm", "init_step", "adapt_step"]),
    ("torchdyn/numerics/odeint.py", ["odeint", "_adaptive_odeint"]),
    ("torchdyn/core/defunc.py", ["DEFunc", "DEFuncBase"]),
]


def load_definitions(path, names, namespace):
    module = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    found = {node.name for node in selected}
    if found != set(names):
        raise ValueError(f"Missing definitions in {path}: {set(names)-found}")
    module.body = selected
    exec(compile(module, str(path), "exec"), namespace)
    return [{"name": node.name, "line_start": node.lineno, "line_end": node.end_lineno}
            for node in selected]


EXTRACTED = []
for relative_path, names in SELECTIONS:
    locations = load_definitions(SOURCE / relative_path, names, globals())
    EXTRACTED.append({"source": relative_path, "definitions": locations})


def integrate(vector_field, initial, times, atol, rtol):
    """Exact plain forward numerical path used by NeuralODE.trajectory.

Passing a solver instance restricts usage to dopri5; event, fixed-step, and
interpolation branches are intentionally not made available in this adapter.
"""
    solver = DormandPrince45(dtype=initial.dtype)
    wrapped = DEFunc(vector_field, order=1)
    wrapped.sensitivity = "adjoint"
    # Original trajectory directly calls odeint and performs no adjoint solve.
    result_times, solution = odeint(wrapped, initial, times, solver,
                                   atol=atol, rtol=rtol)
    return result_times, solution, int(wrapped.nfe)
