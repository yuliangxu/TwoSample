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

def hairer_norm(tensor):
    return tensor.abs().pow(2).mean().sqrt()


def init_step(f, f0, x0, t0, order, atol, rtol):
    scale = atol + torch.abs(x0) * rtol
    d0, d1 = hairer_norm(x0 / scale), hairer_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=t0.dtype, device=t0.device)
    else:
        h0 = 0.01 * d0 / d1

    x_new = x0 + h0 * f0
    f_new = f(t0 + h0, x_new)
    d2 = hairer_norm((f_new - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=t0.dtype, device=t0.device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))
    dt = torch.min(100 * h0, h1).to(t0)
    return dt


@torch.no_grad()
def adapt_step(dt, error_ratio, safety, min_factor, max_factor, order):
    if error_ratio == 0: return dt * max_factor
    if error_ratio < 1: min_factor = torch.ones_like(dt)
    exponent = torch.tensor(order, dtype=dt.dtype, device=dt.device).reciprocal()
    factor = torch.min(max_factor, torch.max(safety / error_ratio ** exponent, min_factor))
    return dt * factor
