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

class DormandPrince45(DiffEqSolver):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=5)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.tableau = construct_dopri5(self.dtype)

    def step(self, f, x, t, dt, k1=None, args=None) -> Tuple:
        c, a, bsol, berr = self.tableau
        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * a[0] * k1)
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
        k5 = f(t + c[3] * dt, x + dt * a[3][0] * k1 + dt * a[3][1] * k2 + dt * a[3][2] * k3 + dt * a[3][3] * k4)
        k6 = f(t + c[4] * dt, x + dt * a[4][0] * k1 + dt * a[4][1] * k2 + dt * a[4][2] * k3 + dt * a[4][3] * k4 + dt * a[4][4] * k5)
        k7 = f(t + c[5] * dt, x + dt * a[5][0] * k1 + dt * a[5][1] * k2 + dt * a[5][2] * k3 + dt * a[5][3] * k4 + dt * a[5][4] * k5 + dt * a[5][5] * k6)
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        err = dt * (berr[0] * k1 + berr[1] * k2 + berr[2] * k3 + berr[3] * k4 + berr[4] * k5 + berr[5] * k6 + berr[6] * k7)
        return k7, x_sol, err, (k1, k2, k3, k4, k5, k6, k7)


class Tsitouras45(DiffEqSolver):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=5)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.tableau = construct_tsit5(self.dtype)

    def step(self, f, x, t, dt, k1=None, args=None) -> Tuple:
        c, a, bsol, berr = self.tableau
        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * a[0][0] * k1)
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
        k5 = f(t + c[3] * dt, x + dt * a[3][0] * k1 + dt * a[3][1] * k2 + dt * a[3][2] * k3 + dt * a[3][3] * k4)
        k6 = f(t + c[4] * dt, x + dt * a[4][0] * k1 + dt * a[4][1] * k2 + dt * a[4][2] * k3 + dt * a[4][3] * k4 + dt * a[4][4] * k5)
        k7 = f(t + c[5] * dt, x + dt * a[5][0] * k1 + dt * a[5][1] * k2 + dt * a[5][2] * k3 + dt * a[5][3] * k4 + dt * a[5][4] * k5 + dt * a[5][5] * k6)
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        err = dt * (berr[0] * k1 + berr[1] * k2 + berr[2] * k3 + berr[3] * k4 + berr[4] * k5 + berr[5] * k6 + berr[6] * k7)
        return k7, x_sol, err, (k1, k2, k3, k4, k5, k6, k7)
