# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Selected definitions from torchdyn 1.0.6; see ../SOURCE.json.
# Loaded by AST to avoid unrelated optional package dependencies.

class DiffEqSolver(nn.Module):
    def __init__(
            self,
            order,
            stepping_class:str="fixed",
            min_factor:float=0.2,
            max_factor:float=10,
            safety:float=0.9
        ):

        super(DiffEqSolver, self).__init__()
        self.order = order
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])
        self.tableau = None
        self.stepping_class = stepping_class

    def sync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device with compatible dtypes"
        device = x.device
        if self.tableau is not None:
            c, a, bsol, berr = self.tableau
            self.tableau = c.to(x), [a.to(x) for a in a], bsol.to(x), berr.to(x)
        t_span = t_span.to(device)
        self.safety = self.safety.to(device)
        self.min_factor = self.min_factor.to(device)
        self.max_factor = self.max_factor.to(device)
        return x, t_span

    def step(self, f, x, t, dt, k1=None, args=None):
        raise NotImplementedError("Stepping rule not implemented for the solver")
